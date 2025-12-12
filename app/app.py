import matplotlib
matplotlib.use('Agg')
import gradio as gr
import os
import sys
import shutil
import librosa
import matplotlib.pyplot as plt
import numpy as np
import math
import subprocess
import tempfile
import cv2
import librosa.display
import random
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from scenedetect import scene_manager as sm, open_video
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.scene_manager import SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg

last_scene_list = []
last_beat_times = None
last_audio_duration = 0.0

def _initialize_backends():
    try:
        fig = plt.figure(figsize=(0.1, 0.1))
        plt.plot([])
        plt.close(fig)
        dummy_audio = np.zeros(2205)
        librosa.stft(y=dummy_audio)
        librosa.beat.beat_track(y=dummy_audio, sr=22050)
    except Exception as exc:
        print("Backend warm-up skipped:", exc)
        raise

_initialize_backends()

def _ffprobe_duration(path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return float(result.stdout.strip())

def _detect_scenes_segment(video_path, start_time, end_time, detector_type, threshold, min_scene_len, frame_window, min_content_val):
    video = open_video(video_path)
    try:
        fps = video.frame_rate
        if start_time and start_time > 0:
            start_tc = FrameTimecode(timecode=start_time, fps=fps)
            try:
                video.seek(start_tc)
            except Exception:
                pass
        duration_tc = None
        if end_time is not None:
            segment_duration = max(0.0, end_time - (start_time or 0.0))
            duration_tc = FrameTimecode(timecode=segment_duration, fps=fps)
            
        scene_manager = SceneManager()
        scene_manager.auto_downscale = True # Use default efficient behavior inside workers
        
        min_scene_len_frames = int(min_scene_len * fps) # Calculate frames from seconds
        
        if detector_type == "detect-threshold":
            scene_manager.add_detector(ThresholdDetector(threshold=threshold, min_scene_len=min_scene_len_frames))
        elif detector_type == "detect-adaptive":
            scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=min_scene_len_frames, frame_window=frame_window, min_content_val=min_content_val))
        else:
            scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len_frames))
            
        scene_manager.detect_scenes(video=video, duration=duration_tc)
        return scene_manager.get_scene_list()
    finally:
        if hasattr(video, "close"):
            try:
                video.close()
            except Exception:
                pass

def _merge_scene_lists(scene_lists):
    merged = []
    for scenes in scene_lists:
        merged.extend(scenes)
    merged.sort(key=lambda x: (x[0].get_seconds(), x[0].get_frames()))
    return merged

def _collect_scene_list(video_path, detector_type, threshold, min_scene_len, frame_window, min_content_val, progress_callback=None):
    progress_description = getattr(sm, "PROGRESS_BAR_DESCRIPTION", "  Detected: %d | Progress")

    def report(value, desc):
        if progress_callback:
            progress_callback(value, desc=desc)

    report(0.05, desc="Preparing video")
    print(f"Detecting scenes using {detector_type} (Threshold: {threshold}, Min Len: {min_scene_len}s)...")
    report(0.2, desc="Initializing parallel detectors")
    cv2.setNumThreads(0) # Disable OpenCV internal threading to avoid conflicts

    video = open_video(video_path)
    total_duration = video.duration.get_seconds() if video.duration else 0
    if hasattr(video, "close"):
        try:
            video.close()
        except Exception:
            pass

    if total_duration == 0:
        segments = [(0, None)]
    else:
        # Split into segments for parallel processing
        # Use more workers for better speed, but limit to reasonable number
        num_segments = min(8, max(1, int(total_duration // 10)))
        segment_duration = total_duration / num_segments
        segments = []
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration if i < num_segments - 1 else total_duration
            segments.append((start_time, end_time))

    scene_lists = []
    # Map segments to future results
    with ProcessPoolExecutor(max_workers=len(segments)) as executor:
        futures = [executor.submit(_detect_scenes_segment, video_path, start, end, detector_type, threshold, min_scene_len, frame_window, min_content_val) for start, end in segments]
        
        count = 0
        for future in as_completed(futures):
            try:
                scene_segments = future.result()
                scene_lists.append(scene_segments)
                
                # Crude progress update based on segments done
                count += 1
                progress_val = 0.2 + (0.6 * (count / len(segments)))
                scene_count = sum(len(scenes) for scenes in scene_lists)
                
                print(f"Segment {count}/{len(segments)} finished. Total scenes found so far: {scene_count}")
                report(progress_val, desc=f"Scanning segment {count}/{len(segments)}...")
            except Exception as e:
                print(f"Error in parallel worker: {e}")

    scene_list = _merge_scene_lists(scene_lists)
    report(0.8, desc="Processing results")
    return scene_list

PRESETS = {
    "Default (CPU Balanced)": {
        "video": ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-r", "30"],
        "audio": ["-c:a", "aac", "-b:a", "192k"],
        "filters": []
    },
    "H264 GPU (High Quality)": {
        "video": [
            "-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", 
            "-rc", "vbr", "-cq", "19", "-multipass", "qres", 
            "-spatial-aq", "1", "-temporal-aq", "1", "-profile:v", "high", 
            "-level", "4.1", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-r", "30"
        ],
        "audio": ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"],
        "filters": []
    },
    "AV1 GPU (Efficient)": {
        "video": ["-c:v", "av1_nvenc", "-preset", "p7", "-cq", "24", "-pix_fmt", "yuv420p", "-r", "30"],
        "audio": ["-c:a", "aac", "-b:a", "320k", "-ar", "48000"],
        "filters": []
    },
    "TEST (Fast/No Resizing)": {
        "video": [
            "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull", 
            "-rc", "constqp", "-qp", "35", "-pix_fmt", "yuv420p", "-r", "30"
        ],
        "audio": ["-c:a", "aac", "-b:a", "96k", "-ar", "44100"],
        "filters": []
    }
}

def render_batch(inputs, output_path, transitions, trans_dur, encoding_args, audio_path=None, specific_transitions=None):
    # Get durations for offset calculation
    durations = [_ffprobe_duration(p) for p in inputs]
    
    inputs_args = []
    for p in inputs:
        inputs_args.extend(["-i", p])
    
    cmd = ["ffmpeg", "-y"] + inputs_args
    
    video_map = "[0:v]"
    
    if len(inputs) > 1:
        filter_complex = []
        current_duration = durations[0]
        
        for i in range(1, len(inputs)):
            offset = current_duration - trans_dur
            if offset < 0: offset = 0
            
            next_v = f"[{i}:v]"
            out_v = f"[v{i}]"
            
            # Pick transition
            if specific_transitions and i-1 < len(specific_transitions):
                trans = specific_transitions[i-1]
            else:
                trans = random.choice(transitions) if transitions else "fade"
            
            filter_complex.append(f"{video_map}{next_v}xfade=transition={trans}:duration={trans_dur}:offset={offset}{out_v}")
            
            video_map = out_v
            # Update duration: Previous Sum + Next Duration - Overlap
            current_duration += durations[i] - trans_dur
        
        # Force pixel format to yuv420p and fps to 30 at end of filter chain
        filter_complex.append(f"{video_map}format=yuv420p,fps=30[vout]")
        video_map = "[ vout]"
            
        cmd.extend(["-filter_complex", ";".join(filter_complex)])
    else:
        # Single input pass-through (rare in batching but possible for remainders)
        pass

    if audio_path:
        cmd.extend(["-i", audio_path])
        # Audio is the last input
        audio_idx = len(inputs)
        cmd.extend(["-map", video_map])
        cmd.extend(["-map", f"{audio_idx}:a"])
        
        # Use preset audio args
        if "audio" in encoding_args:
            cmd.extend(encoding_args["audio"])
        else:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
            
        cmd.append("-shortest")
    else:
        cmd.extend(["-map", video_map])
        cmd.extend(["-an"]) # No audio for intermediate batches

    # Use preset video args
    if "video" in encoding_args:
        cmd.extend(encoding_args["video"])
    else:
         cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"])

    cmd.append(output_path)
    
    print(f"Rendering batch: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, capture_output=True)

def _unused_create_amv(video_file, audio_file, cut_intensity, video_direction, shuffle_scenes, allow_stretch, selected_transitions, quality_preset, clips_speed, crop_start, crop_end, aspect_ratio, selected_clips):
    global last_scene_list, last_beat_times, last_audio_duration
    
    TRANSITIONS = {
        "None": None,
        "Flash Blanco (Fade White)": "fadewhite",
        "Zoom In (Zoom In)": "zoomin",
        "Radar (Radial)": "radial",
        "Deslizar Izq (Slide Left)": "slideleft",
        "Deslizar Der (Slide Right)": "slideright",
        "Barrido (Wipe Right)": "wiperight",
        "Elástico (Smooth Left)": "smoothleft",
        "Pixelar (Pixelize)": "pixelize",
        "Desenfoque (H-Blur)": "hblur",
        "Aplastar (Squeeze V)": "squeezev",
        "Círculo (Circle Open)": "circleopen"
    }
    
    encoding_params = PRESETS.get(quality_preset, PRESETS["Default (CPU Balanced)"])
    
    # Filter valid transitions from selection
    active_transitions = []
    if selected_transitions:
        for t in selected_transitions:
            code = TRANSITIONS.get(t)
            if code:
                active_transitions.append(code)
    
    TRANS_DUR = 0.3

    if not video_file or not audio_file:
        print("Video or audio file is missing.")
        return None, "Error: Missing files."

    if not last_scene_list:
        print("No scenes detected yet. Run 'Generate Clips' first.")
        return None, "Error: No scenes detected."
    if last_beat_times is None or len(last_beat_times) < 2:
        print("No beat information available. Run 'Detect Rhythm' first.")
        return None, "Error: No rhythm detected."

    video_path = video_file.name
    original_audio_path = audio_file.name
    audio_path = original_audio_path

    # Handle Audio Cropping
    temp_audio_file = None
    if crop_start > 0 or (crop_end > 0 and crop_end > crop_start):
        print(f"Cropping audio: {crop_start}s to {crop_end if crop_end > 0 else 'End'}s")
        temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_audio_fd)
        temp_audio_file = temp_audio_path
        
        cmd_crop = ["ffmpeg", "-y", "-i", original_audio_path, "-ss", str(crop_start)]
        if crop_end > 0:
            cmd_crop.extend(["-to", str(crop_end)])
        cmd_crop.extend(["-c:a", "pcm_s16le", temp_audio_path]) # Convert to wav for consistency
        
        try:
            subprocess.run(cmd_crop, check=True, capture_output=True)
            audio_path = temp_audio_path
        except subprocess.CalledProcessError as e:
            print(f"Error cropping audio: {e}")
            return None, f"Error cropping audio: {str(e)}"

    print(f"Video path: {video_path}")
    print(f"Audio path: {audio_path}")

    try:
        total_audio_duration = librosa.get_duration(path=audio_path)
        last_audio_duration = total_audio_duration

        # Filter scenes based on selection
        if selected_clips is not None:
             if len(selected_clips) == 0:
                 scenes = [] # Explicitly empty if user deselected all
             else:
                 indices = []
                 for s in selected_clips:
                     try:
                         # Parse "Clip X (...)" to get X
                         # Expecting "Clip 1 (00:00 - 00:05)"
                         idx_str = s.split(" ")[1]
                         idx = int(idx_str) - 1
                         indices.append(idx)
                     except:
                         pass
                 scenes = [last_scene_list[i] for i in indices if 0 <= i < len(last_scene_list)]
        else:
            scenes = list(last_scene_list)
            
        if not scenes:
            print("No scenes selected.")
            return None, "Error: No scenes selected. Please select at least one clip."

        if shuffle_scenes:
            random.shuffle(scenes)

        segments = []
        scene_idx = 0

        # Group beats based on cut_intensity
        beat_groups = []
        
        # Handle Intro (time before first beat)
        first_beat_time = last_beat_times[0]
        has_intro = False
        if first_beat_time > 0.1:
            beat_groups.append({"duration": first_beat_time, "type": "Intro"})
            has_intro = True

        current_beat_idx = 0
        while current_beat_idx < len(last_beat_times) - 1:
            next_idx = min(current_beat_idx + int(cut_intensity), len(last_beat_times) - 1)
            start_time = last_beat_times[current_beat_idx]
            end_time = last_beat_times[next_idx]
            beat_groups.append({"duration": end_time - start_time, "type": "Beat"})
            current_beat_idx = next_idx

        for group_idx, group in enumerate(beat_groups):
            target_dur = float(group["duration"])
            segment_type = group["type"]
            
            if target_dur < 0.1:
                continue
                
            best_match = None
            search_attempts = 0
            
            # Target output duration for this segment (what we want in the AMV)
            final_segment_dur = target_dur
            
            # Adjust required source duration based on speed
            # If speed is 2.0x, we need 2x the source duration to fill the same time gap?
            # No, if we want 2 seconds of output at 2x speed, we need 4 seconds of source.
            # If we want 2 seconds of output at 0.5x speed, we need 1 second of source.
            required_src_dur_for_beat = final_segment_dur * clips_speed
            
            # Add transition padding to the *output* duration we are aiming for
            # But wait, padding needs to be available in the source too.
            # Actually, we extract `required_src` and then speed-change it to `final_segment_dur`.
            # If we have transitions, the *output* clip needs to be longer by TRANS_DUR.
            # So we need (TRANS_DUR * clips_speed) more source time.
            
            output_dur_needed = final_segment_dur
            if active_transitions and group_idx > 0:
                 output_dur_needed += TRANS_DUR
                 
            required_src_dur = output_dur_needed * clips_speed

            while search_attempts < len(scenes):
                idx = (scene_idx + search_attempts) % len(scenes)
                start_tc, end_tc = scenes[idx]
                dur = end_tc.get_seconds() - start_tc.get_seconds()
                if dur >= required_src_dur:
                    best_match = (idx, start_tc.get_seconds(), dur)
                    break
                search_attempts += 1
                
            # Determine direction for this segment
            current_direction = video_direction
            if video_direction == "Random":
                current_direction = random.choice(["Forward", "Backward"])

            if best_match:
                idx, start, dur = best_match
                segments.append({
                    "mode": "normal",
                    "start": start,
                    "src_duration": required_src_dur, # Extract exactly what we need
                    "target_duration": final_segment_dur,
                    "direction": current_direction,
                    "type": segment_type,
                    "speed": clips_speed
                })
                scene_idx = (idx + 1) % len(scenes)
            else:
                # Fallback: Stretch or Loop (Using longest clip)
                longest_idx = 0
                max_d = 0
                for i_s, (st_tc, en_tc) in enumerate(scenes):
                    d_s = en_tc.get_seconds() - st_tc.get_seconds()
                    if d_s > max_d:
                        max_d = d_s
                        longest_idx = i_s
                
                idx = longest_idx
                start_tc, end_tc = scenes[idx]
                dur = end_tc.get_seconds() - start_tc.get_seconds()
                dur = max(dur, 0.1)
                
                # If standard stretch is allowed (to fit beat)
                # If clips_speed is 1.0, normal stretch logic applies.
                # If clips_speed is != 1.0, we are already applying a speed effect.
                # In fallback, we just want to fill the gap.
                
                mode = "loop"
                # If we have enough source to cover at least half the requirement, stretch it
                if allow_stretch and dur >= required_src_dur * 0.5:
                    mode = "stretch"
                
                segments.append({
                    "mode": mode,
                    "start": start_tc.get_seconds(),
                    "src_duration": dur, # Use full available duration
                    "target_duration": final_segment_dur,
                    "direction": current_direction,
                    "type": segment_type,
                    "speed": clips_speed
                })
                scene_idx = (scene_idx + 1) % len(scenes)

        # Fill remaining time if any
        current_gen_duration = sum(s["target_duration"] for s in segments)
        if current_gen_duration < total_audio_duration:
            remaining = total_audio_duration - current_gen_duration
            if remaining > 0.1:
                 idx = scene_idx % len(scenes)
                 start_tc, end_tc = scenes[idx]
                 dur = end_tc.get_seconds() - start_tc.get_seconds()

                 current_direction = video_direction
                 if video_direction == "Random":
                    current_direction = random.choice(["Forward", "Backward"])
                 
                 req_rem = remaining
                 if active_transitions and len(segments) > 0:
                     req_rem += TRANS_DUR

                 segments.append({
                     "mode": "loop",
                     "start": start_tc.get_seconds(),
                     "src_duration": dur,
                     "target_duration": remaining,
                     "direction": current_direction,
                     "type": "Outro/Fill",
                     "speed": clips_speed
                 })

        if not segments:
            print("No suitable scenes matched the beats.")
            return None, "No suitable scenes matched the beats."

        # Generate synchronization log
        sync_log = ["Beat Sync Report:", "-----------------"]
        cumulative_time = 0.0
        for i, seg in enumerate(segments):
            start_time = cumulative_time
            end_time = cumulative_time + seg["target_duration"]
            dir_arrow = "->" if seg["direction"] == "Forward" else "<-"
            seg_type = seg.get("type", "Beat")
            speed_lbl = f"{seg['speed']}x"
            sync_log.append(f"Cut {i+1} [{seg_type}]: {start_time:.2f}s - {end_time:.2f}s | {seg['mode'].title()} | {dir_arrow} {seg['direction']} | Spd: {speed_lbl}")
            cumulative_time = end_time
        sync_report = "\n".join(sync_log)

        tmpdir = tempfile.mkdtemp()
        clip_paths = []
        try:
            for idx, seg in enumerate(segments):
                clip_path = os.path.join(tmpdir, f"clip_{idx+1:03}.mp4")
                
                # Target output duration of this clip file
                t_dur = seg["target_duration"]
                if active_transitions and idx > 0:
                    t_dur += TRANS_DUR
                
                # Fade Logic
                vf_fade = ""
                if not active_transitions:
                    fade_dur = max(0.1, min(0.5, t_dur / 5.0))
                    fade_out_st = t_dur - fade_dur
                    if fade_out_st < 0: fade_out_st = 0
                    vf_fade = f"fade=t=in:st=0:d={fade_dur},fade=t=out:st={fade_out_st}:d={fade_dur}"
                
                # Base args using preset video params
                base_args = encoding_params["video"] + ["-an", "-threads", "4"]
                
                # Filter Chain Construction
                filter_chain = []
                
                # 1. SetPTS (Speed / Stretch)
                # Standard formula for speed change: setpts=(1/speed)*PTS
                # If "stretch" mode is active (fallback), we need to calculate specific factor
                # to force src_duration to match t_dur exactly.
                if seg["mode"] == "stretch":
                    # Force fit
                    # factor = target / source
                    # But SetPTS multiplier is inverse of speed? 
                    # setpts=2.0*PTS means SLOWER (duration doubles)
                    # setpts=0.5*PTS means FASTER (duration halves)
                    # Factor = TargetDuration / SourceDuration
                    factor = t_dur / seg["src_duration"]
                    filter_chain.append(f"setpts={factor}*PTS")
                else:
                    # Apply user defined speed
                    # If user wants 2.0x speed (fast), multiplier is 0.5
                    pts_mult = 1.0 / seg["speed"]
                    filter_chain.append(f"setpts={pts_mult}*PTS")

                # 2. Reverse (If needed)
                if seg["direction"] == "Backward":
                    filter_chain.append("reverse")
                
                # 3. Aspect Ratio Cropping (High Quality Scaling)
                if aspect_ratio == "3:4 (TikTok/Shorts)":
                    # Target 1080x1440 (Scale height to 1440, crop center 1080)
                    filter_chain.append("scale=-1:1440:flags=lanczos,crop=1080:1440:(iw-1080)/2:0")
                elif aspect_ratio == "9:16 (Vertical)":
                    # Target 1080x1920
                    filter_chain.append("scale=-1:1920:flags=lanczos,crop=1080:1920:(iw-1080)/2:0")
                
                # 4. Fade (Only if no xfade)
                if vf_fade:
                    filter_chain.append(vf_fade.lstrip(','))
                
                # 4. Preset Filters (e.g., Scale)
                if encoding_params.get("filters"):
                    filter_chain.extend(encoding_params["filters"])
                
                # Force constant frame rate for individual clips too
                filter_chain.append("fps=30")
                
                vf_string = ",".join(filter_chain) if filter_chain else "null"

                # Single-step FFmpeg command
                input_args = []
                # Use 2-step for loop/backward stability
                if seg["mode"] == "loop" or seg["direction"] == "Backward":
                    raw_clip = os.path.join(tmpdir, f"raw_{idx}.mp4")
                    cmd_extract = [
                        "ffmpeg", "-y",
                        "-ss", f"{seg['start']:.3f}",
                        "-t", f"{seg['src_duration']:.6f}",
                        "-i", video_path,
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-an", "-threads", "4",
                        raw_clip
                    ]
                    subprocess.run(cmd_extract, check=True, capture_output=True)
                    
                    input_args = ["-i", raw_clip]
                    if seg["mode"] == "loop":
                        input_args = ["-stream_loop", "-1"] + input_args

                    cmd_proc = [
                        "ffmpeg", "-y"
                    ] + input_args + [
                        "-vf", vf_string,
                        "-t", f"{t_dur:.6f}"
                    ] + base_args + [clip_path]
                    subprocess.run(cmd_proc, check=True, capture_output=True)
                    
                else:
                    cmd_proc = [
                        "ffmpeg", "-y",
                        "-ss", f"{seg['start']:.3f}",
                        "-t", f"{seg['src_duration']:.6f}",
                        "-i", video_path,
                        "-vf", vf_string
                    ] + base_args + [
                        "-t", f"{t_dur:.6f}", 
                        clip_path
                    ]
                    subprocess.run(cmd_proc, check=True, capture_output=True)
                
                clip_paths.append(clip_path)

            output_video = "generated_amv.mp4"
            
            if active_transitions and len(clip_paths) > 1:
                # Batch Processing
                print(f"Applying transitions from: {selected_transitions} (Batch Processing)")
                
                BATCH_SIZE = 4
                current_clips = clip_paths
                iteration = 0
                
                while len(current_clips) > 1:
                    next_clips = []
                    # Calculate number of batches
                    num_batches = math.ceil(len(current_clips) / BATCH_SIZE)
                    
                    if num_batches == 1:
                        print("Final pass rendering...")
                        render_batch(
                            current_clips, 
                            output_video, 
                            active_transitions, 
                            TRANS_DUR, 
                            encoding_params,
                            audio_path=audio_path
                        )
                        break
                    
                    print(f"Batch iteration {iteration}: Processing {num_batches} batches...")
                    for b in range(num_batches):
                        chunk = current_clips[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
                        
                        if len(chunk) == 1:
                            next_clips.append(chunk[0])
                            continue
                            
                        part_name = os.path.join(tmpdir, f"iter_{iteration}_batch_{b}.mp4")
                        render_batch(
                            chunk, 
                            part_name, 
                            active_transitions, 
                            TRANS_DUR, 
                            encoding_params,
                            audio_path=None
                        )
                        next_clips.append(part_name)
                    
                    current_clips = next_clips
                    iteration += 1
                
            else:
                # Standard simple concatenation
                concat_list = os.path.join(tmpdir, "concat.txt")
                with open(concat_list, "w", encoding="utf-8") as f:
                    for path in clip_paths:
                        normalized_path = path.replace("\\\\", "/")
                        f.write(f"file '{normalized_path}'\n")

                concat_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list,
                    "-i", audio_path,
                    "-map", "0:v",
                    "-map", "1:a"
                ]
                
                if "video" in encoding_params:
                     concat_cmd.extend(encoding_params["video"])
                else:
                     concat_cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"])
                     
                if "audio" in encoding_params:
                     concat_cmd.extend(encoding_params["audio"])
                else:
                     concat_cmd.extend(["-c:a", "aac", "-b:a", "192k"])
                
                concat_cmd.append("-shortest")
                concat_cmd.append(output_video)
                
                subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
            print("Successfully created AMV.")
            return output_video, sync_report
        except subprocess.CalledProcessError as e:
            print("Error creating AMV with ffmpeg.")
            print("Stderr:", e.stderr)
            return None, f"Error: {str(e)}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if temp_audio_file and os.path.exists(temp_audio_file):
             os.remove(temp_audio_file)
def preview_audio_cut(audio_file, crop_start, crop_end):
    if audio_file is None:
        return None
        
    original_audio_path = audio_file.name
    
    fd, preview_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    
    cmd_crop = ["ffmpeg", "-y", "-i", original_audio_path, "-ss", str(crop_start)]
    if crop_end > 0 and crop_end > crop_start:
        cmd_crop.extend(["-to", str(crop_end)])
    
    cmd_crop.extend(["-c:a", "pcm_s16le", preview_path])
    
    try:
        subprocess.run(cmd_crop, check=True, capture_output=True)
        return preview_path
    except Exception as e:
        print(f"Preview error: {e}")
        return None

def detect_rhythm(audio_file, algorithm, tightness, crop_start, crop_end, progress=gr.Progress()):
    global last_beat_times
    if audio_file is None:
        return None, ""
    
    audio_path = audio_file.name
    print(f"Loading audio for rhythm detection (Algorithm: {algorithm})...")
    progress(0, desc="Loading Audio")
    
    try:
        duration = None
        if crop_end > 0 and crop_end > crop_start:
            duration = crop_end - crop_start
            
        y, sr = librosa.load(audio_path, offset=crop_start, duration=duration)
        print("Audio loaded.")
    
        print("Tracking beat...")
        progress(0.25, desc=f"Tracking Beat ({algorithm})")
    
        if algorithm == "Low Percussion":
            # Optimized for music without clear percussion (Piano, Violin, Ambient)
            C = np.abs(librosa.cqt(y=y, sr=sr))
            onset_env = librosa.onset.onset_strength(sr=sr, S=librosa.amplitude_to_db(C, ref=np.max))
            tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, tightness=tightness)
        else:
            # Standard (Percussive)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, tightness=tightness)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        print("Beat tracking complete.")
        last_beat_times = list(beat_times)
        
        # Handle tempo being a numpy array or float
        bpm_val = float(tempo) if np.ndim(tempo) == 0 else float(tempo[0])
        
        if beat_times.size:
            first_beat = beat_times[0]
            last_beat = beat_times[-1]
            info_str = (
                f"Algorithm: {algorithm} | BPM Est: {bpm_val:.2f}\n"
                f"Beats detected: {len(beat_times)}\n"
                f"First beat: {first_beat:.3f}s\n"
                f"Last beat: {last_beat:.3f}s\n\n"
                """--- Beat Segments ---
"""
            )
            for i in range(len(beat_times) - 1):
                start = beat_times[i]
                end = beat_times[i+1]
                dur = end - start
                info_str += f"Segment {i+1}: {start:.3f}s - {end:.3f}s (Duration: {dur:.3f}s)\n"
        else:
            info_str = "Beats detected: 0"
        
        print("Generating plot...")
        progress(0.75, desc="Generating Plot")
        fig, ax = plt.subplots(figsize=(15, 2))
        librosa.display.waveshow(y, sr=sr, alpha=0.5, ax=ax)
        ax.vlines(beat_times, -1, 1, color='r')
        ax.set_ylim(-1, 1)
        ax.set_title(f'Beat Tracker ({algorithm})')
        print("Plot generated.")
        
        progress(1, desc="Done")
        return fig, info_str
        
    except Exception as e:
        print(f"Error in rhythm detection: {e}")
        return None, f"Error: {str(e)}"


def generate_previews(video_path, scene_list, mode="Thumbnails"):
    tmp_dir = tempfile.mkdtemp()
    previews = []
    
    print(f"Generating previews ({mode}) for {len(scene_list)} scenes...")
    
    if mode == "Thumbnails":
        cap = cv2.VideoCapture(video_path)
        for i, (start, end) in enumerate(scene_list):
            try:
                frame_num = start.get_frames() + 5 
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start.get_frames())
                    ret, frame = cap.read()
                    
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img_path = os.path.join(tmp_dir, f"thumb_{i}.jpg")
                    cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    duration = end.get_seconds() - start.get_seconds()
                    label = f"#{i+1} ({duration:.1f}s)"
                    previews.append((img_path, label))
                else:
                    previews.append(None)
            except Exception as e:
                print(f"Error extracting thumbnail for scene {i}: {e}")
                previews.append(None)
        cap.release()
        
    else:
        # Video Split Mode
        if "Copy" in mode:
            args = "-c:v copy -c:a copy"
        else:
            args = "-c:v libx264 -preset fast -crf 23 -c:a aac"
            
        try:
            split_video_ffmpeg(
                video_path, 
                scene_list, 
                output_file_template=os.path.join(tmp_dir, "preview_$SCENE_NUMBER.mp4"),
                arg_override=args,
                show_progress=True
            )
            
            # Collect generated files
            # scenedetect uses 001, 002 format for scene numbers
            for i, (start, end) in enumerate(scene_list):
                # We need to match the filename generated by scenedetect
                # It typically pads numbers. Let's assume 3 digits as default or try to find it.
                # Actually, strictly following the template "preview_$SCENE_NUMBER.mp4"
                # Scene 1 -> preview_001.mp4 usually
                
                # Let's glob or just predict
                # Scenedetect by default uses 3 digits padding for $SCENE_NUMBER
                vid_path = os.path.join(tmp_dir, f"preview_{i+1:03d}.mp4")
                
                if os.path.exists(vid_path):
                    duration = end.get_seconds() - start.get_seconds()
                    label = f"#{i+1} ({duration:.1f}s)"
                    previews.append((vid_path, label))
                else:
                    previews.append(None)
                    
        except Exception as e:
            print(f"Error splitting video for preview: {e}")
            return []

    return previews

def srt_time_to_ass(time_str):
    # 00:00:00,900 -> 0:00:00.90
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    cs = int(ms) // 10
    return f"{int(h)}:{m}:{s}.{cs:02d}"

def convert_srt_to_ass(srt_path, ass_path):
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by blocks
    blocks = re.split(r'\n\n+', content.strip())
    
    # Check if this is a "Karaoke/Highlight" SRT (contains <u> tags)
    is_karaoke = '<u>' in content
    
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Impact,80,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,10,10,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    
    for block in blocks:
        lines = block.split('\n')
        if len(lines) < 3:
            continue
            
        # Parse timestamp: 00:00:00,900 --> 00:00:01,580
        times = lines[1].split(' --> ')
        if len(times) != 2:
            continue
            
        start = srt_time_to_ass(times[0].strip())
        end = srt_time_to_ass(times[1].strip())
        text = "\n".join(lines[2:])
        
        # Logic for Karaoke SRTs
        if is_karaoke:
            if '<u>' in text:
                # Active line: Transform <u>Word</u> into Highlight Style
                # Cyan Color (&H00FFFF& -> BGR -> FFFF00) + Scale 115%
                # Use standard ASS override tags
                text = re.sub(r'<u>(.*?)</u>', r'{\\c&H00FFFF&\\fscx115\\fscy115}\1{\\fscx100\\fscy100\\c&HFFFFFF&}', text)
                events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
            else:
                # Non-active line in a karaoke file (likely a summary line)
                # Ignore it to avoid double rendering
                continue
        else:
            # Standard SRT - just render
            events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(header + "\n".join(events))

def burn_subtitles(video_file, srt_file):
    if not video_file or not srt_file:
        return None, "Error: Missing video or subtitle file."
        
    video_path = video_file.name
    srt_path = srt_file.name
    
    # Stage everything locally to avoid FFmpeg Windows path escaping hell
    local_video = "temp_input.mp4"
    local_ass = "temp_subs.ass"
    output_path = "video_with_subs.mp4"
    
    # Clean up previous runs
    for f in [local_video, local_ass, output_path]:
        if os.path.exists(f):
            os.remove(f)
            
    try:
        print(f"Copying video to local dir: {local_video}...")
        shutil.copy(video_path, local_video)
        
        print("Converting SRT to ASS dynamic format...")
        convert_srt_to_ass(srt_path, local_ass)
        
        print("Burning subtitles with FFmpeg (Local mode)...")
        print(f"FFmpeg path: {shutil.which('ffmpeg')}")
        
        # Use simple filenames since we are in the same dir
        cmd = [
            "ffmpeg", "-y",
            "-i", local_video,
            "-vf", f"subtitles=filename={local_ass}",
            "-c:a", "copy",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            output_path
        ]
        
        print(f"Executing FFmpeg command: {cmd}")
        subprocess.run(cmd, check=True, capture_output=True, cwd=os.getcwd())
        return output_path, "Subtitles burned successfully.\nMode: " + ("Dynamic Karaoke" if '<u>' in open(srt_path).read() else "Standard"), local_ass
        
    except Exception as e:
        print(f"Error: {e}")
        # Try to read stderr if it's a subprocess error
        if isinstance(e, subprocess.CalledProcessError):
             err_msg = e.stderr.decode('utf-8', errors='ignore')
             print(f"FFmpeg Stderr: {err_msg}")
             
             # FALLBACK: Check if error is due to missing libass filter
             if "No such filter" in err_msg and ("ass" in err_msg or "subtitles" in err_msg):
                 print("\n[Auto-Fix] System FFmpeg lacks libass/subtitles support. Attempting fallback to 'imageio-ffmpeg'...")
                 try:
                     try:
                         import imageio_ffmpeg
                     except ImportError:
                         print("Installing imageio-ffmpeg package...")
                         subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio-ffmpeg"])
                         import imageio_ffmpeg
                     
                     fallback_exe = imageio_ffmpeg.get_ffmpeg_exe()
                     print(f"Fallback FFmpeg found at: {fallback_exe}")
                     
                     # Update command to use fallback executable
                     cmd[0] = fallback_exe
                     # Retry execution
                     print(f"Retrying with fallback FFmpeg: {cmd}")
                     subprocess.run(cmd, check=True, capture_output=True, cwd=os.getcwd())
                     return output_path, "Subtitles burned (Fallback FFmpeg).\nMode: " + ("Dynamic Karaoke" if '<u>' in open(srt_path).read() else "Standard"), local_ass
                     
                 except Exception as fallback_err:
                     print(f"Fallback failed: {fallback_err}")
                     return None, f"Error (System FFmpeg missing filters, Fallback failed): {str(fallback_err)}", None

        return None, f"Error: {str(e)}", None
    finally:
        # Cleanup temps
        if os.path.exists(local_video):
            os.remove(local_video)
        # Keep local_ass for download

def apply_video_filter(video_file, preset, cas, contrast, saturation, brightness, gamma):
    if not video_file:
        return None, "Error: No video file."
    
    video_path = video_file.name
    output_path = "video_enhanced.mp4"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Defaults
    vf_chain = ""
    
    if preset == "CLEAN (Natural/Cine)":
        # cas=0.3,eq=contrast=1.05:saturation=1.1:brightness=0.0:gamma=1.0
        vf_chain = "cas=0.3,eq=contrast=1.05:saturation=1.1:brightness=0.0:gamma=1.0"
    elif preset == "VIBRANT (Anime/Standard)":
        # cas=0.6,eq=contrast=1.12:saturation=1.3:brightness=0.02:gamma=1.05
        vf_chain = "cas=0.6,eq=contrast=1.12:saturation=1.3:brightness=0.02:gamma=1.05"
    elif preset == "AGGRESSIVE (Max/Epic)":
        # cas=0.9,eq=contrast=1.25:saturation=1.5:brightness=-0.05:gamma=1.2
        vf_chain = "cas=0.9,eq=contrast=1.25:saturation=1.5:brightness=-0.05:gamma=1.2"
    else: # Custom
        vf_chain = f"cas={cas},eq=contrast={contrast}:saturation={saturation}:brightness={brightness}:gamma={gamma}"
        
    try:
        print(f"Applying filter preset: {preset}")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-map_metadata", "-1", # Clear metadata
            "-metadata", f"title=AutoAMV_{timestamp}",
            "-metadata", "artist=ChronAIclesMusic",
            "-vf", vf_chain,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "copy", # Copy audio
            output_path
        ]
        
        print(f"Command: {cmd}")
        subprocess.run(cmd, check=True, capture_output=True, cwd=os.getcwd())
        return output_path, f"Filter '{preset}' applied successfully."
        
    except Exception as e:
        print(f"Error: {e}")
        if isinstance(e, subprocess.CalledProcessError):
             err_msg = e.stderr.decode('utf-8', errors='ignore')
             print(f"FFmpeg Stderr: {err_msg}")
             return None, f"Error: {err_msg}"
        return None, f"Error: {str(e)}"


def detect_scenes(video_file, detector_type, threshold, min_scene_len, frame_window, min_content_val, preview_mode="Thumbnails", progress=gr.Progress()):
    global last_scene_list
    if video_file is None:
        return [], gr.update(choices=[], value=[])
    
    video_path = video_file.name
    progress(0.05, desc="Preparing video")
    
    scene_list = _collect_scene_list(video_path, detector_type, threshold, min_scene_len, frame_window, min_content_val, progress_callback=progress)
    progress(0.8, desc="Processing results")
    if not scene_list:
        progress(1, desc="Done")
        print("No scenes detected.")
        return [], gr.update(choices=[], value=[])
        
    last_scene_list = scene_list
    
    choices = []
    
    for i in range(len(scene_list)):
        start, end = scene_list[i]
        duration = end.get_seconds() - start.get_seconds()
        # "Clip 1 (00:00:00 - 00:00:05) [5.00s]"
        label = f"Clip {i+1} ({start.get_timecode()} - {end.get_timecode()}) [{duration:.2f}s]"
        choices.append(label)
        

    print(f"Generating previews ({preview_mode})...")
    progress(0.9, desc=f"Generating previews ({preview_mode})")
    previews = generate_previews(video_path, scene_list, mode=preview_mode)

    print("Scene detection complete.")
    progress(1, desc="Done")
    
    return previews, gr.update(choices=choices, value=choices)

def create_amv(video_file, audio_file, cut_intensity, video_direction, shuffle_scenes, allow_stretch, selected_transitions, quality_preset, clips_speed, crop_start, crop_end, aspect_ratio, selected_clips):
    global last_scene_list, last_beat_times, last_audio_duration
    
    TRANSITIONS = {
        "None": None,
        "Flash Blanco (Fade White)": "fadewhite",
        "Zoom In (Zoom In)": "zoomin",
        "Radar (Radial)": "radial",
        "Deslizar Izq (Slide Left)": "slideleft",
        "Deslizar Der (Slide Right)": "slideright",
        "Barrido (Wipe Right)": "wiperight",
        "Elástico (Smooth Left)": "smoothleft",
        "Pixelar (Pixelize)": "pixelize",
        "Desenfoque (H-Blur)": "hblur",
        "Aplastar (Squeeze V)": "squeezev",
        "Círculo (Circle Open)": "circleopen"
    }
    
    encoding_params = PRESETS.get(quality_preset, PRESETS["Default (CPU Balanced)"])
    
    # Filter valid transitions from selection
    active_transitions = []
    if selected_transitions:
        for t in selected_transitions:
            code = TRANSITIONS.get(t)
            if code:
                active_transitions.append(code)
    
    TRANS_DUR = 0.3

    if not video_file or not audio_file:
        print("Video or audio file is missing.")
        return None, "Error: Missing files."

    if not last_scene_list:
        print("No scenes detected yet. Run 'Generate Clips' first.")
        return None, "Error: No scenes detected."
    if last_beat_times is None or len(last_beat_times) < 2:
        print("No beat information available. Run 'Detect Rhythm' first.")
        return None, "Error: No rhythm detected."

    video_path = video_file.name
    original_audio_path = audio_file.name
    audio_path = original_audio_path

    # Handle Audio Cropping
    temp_audio_file = None
    if crop_start > 0 or (crop_end > 0 and crop_end > crop_start):
        print(f"Cropping audio: {crop_start}s to {crop_end if crop_end > 0 else 'End'}s")
        temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_audio_fd)
        temp_audio_file = temp_audio_path
        
        cmd_crop = ["ffmpeg", "-y", "-i", original_audio_path, "-ss", str(crop_start)]
        if crop_end > 0:
            cmd_crop.extend(["-to", str(crop_end)])
        cmd_crop.extend(["-c:a", "pcm_s16le", temp_audio_path]) # Convert to wav for consistency
        
        try:
            subprocess.run(cmd_crop, check=True, capture_output=True)
            audio_path = temp_audio_path
        except subprocess.CalledProcessError as e:
            print(f"Error cropping audio: {e}")
            return None, f"Error cropping audio: {str(e)}"

    print(f"Video path: {video_path}")
    print(f"Audio path: {audio_path}")

    try:
        total_audio_duration = librosa.get_duration(path=audio_path)
        last_audio_duration = total_audio_duration

        # Filter scenes based on selection
        scenes = []
        if selected_clips is not None and len(selected_clips) > 0:
             indices = []
             for s in selected_clips:
                 try:
                     # Parse "Clip X (...)" to get X
                     idx_str = s.split(" ")[1]
                     idx = int(idx_str) - 1
                     indices.append(idx)
                 except:
                     pass
             scenes = [(last_scene_list[i], i+1) for i in indices if 0 <= i < len(last_scene_list)]
        else:
            scenes = [(s, i+1) for i, s in enumerate(last_scene_list)]
            
        if not scenes:
            print("No scenes selected.")
            return None, "Error: No scenes selected. Please select at least one clip."

        if shuffle_scenes:
            random.shuffle(scenes)

        segments = []
        scene_idx = 0

        # Group beats based on cut_intensity
        beat_groups = []
        
        # Handle Intro (time before first beat)
        first_beat_time = last_beat_times[0]
        has_intro = False
        if first_beat_time > 0.1:
            beat_groups.append({"duration": first_beat_time, "type": "Intro"})
            has_intro = True

        current_beat_idx = 0
        while current_beat_idx < len(last_beat_times) - 1:
            next_idx = min(current_beat_idx + int(cut_intensity), len(last_beat_times) - 1)
            start_time = last_beat_times[current_beat_idx]
            end_time = last_beat_times[next_idx]
            beat_groups.append({"duration": end_time - start_time, "type": "Beat"})
            current_beat_idx = next_idx

        for group_idx, group in enumerate(beat_groups):
            target_dur = float(group["duration"])
            segment_type = group["type"]
            
            if target_dur < 0.1:
                continue
                
            best_match = None
            search_attempts = 0
            
            # Target output duration for this segment (what we want in the AMV)
            final_segment_dur = target_dur
            
            # Adjust required source duration based on speed
            required_src_dur_for_beat = final_segment_dur * clips_speed
            
            output_dur_needed = final_segment_dur
            if active_transitions and group_idx > 0:
                 output_dur_needed += TRANS_DUR
                 
            required_src_dur = output_dur_needed * clips_speed

            while search_attempts < len(scenes):
                idx = (scene_idx + search_attempts) % len(scenes)
                (start_tc, end_tc), orig_idx = scenes[idx]
                dur = end_tc.get_seconds() - start_tc.get_seconds()
                if dur >= required_src_dur:
                    best_match = (idx, start_tc.get_seconds(), dur, orig_idx)
                    break
                search_attempts += 1
                
            # Determine direction for this segment
            current_direction = video_direction
            if video_direction == "Random":
                current_direction = random.choice(["Forward", "Backward"])

            if best_match:
                idx, start, dur, orig_idx = best_match
                segments.append({
                    "mode": "normal",
                    "start": start,
                    "src_duration": required_src_dur, # Extract exactly what we need
                    "target_duration": final_segment_dur,
                    "direction": current_direction,
                    "type": segment_type,
                    "speed": clips_speed,
                    "orig_clip_id": orig_idx
                })
                scene_idx = (idx + 1) % len(scenes)
            else:
                # Fallback: Stretch or Loop (Using longest clip)
                longest_idx = 0
                max_d = 0
                for i_s, ((st_tc, en_tc), _) in enumerate(scenes):
                    d_s = en_tc.get_seconds() - st_tc.get_seconds()
                    if d_s > max_d:
                        max_d = d_s
                        longest_idx = i_s
                
                idx = longest_idx
                (start_tc, end_tc), orig_idx = scenes[idx]
                dur = end_tc.get_seconds() - start_tc.get_seconds()
                dur = max(dur, 0.1)
                
                mode = "loop"
                if allow_stretch and dur >= required_src_dur * 0.5:
                    mode = "stretch"
                
                segments.append({
                    "mode": mode,
                    "start": start_tc.get_seconds(),
                    "src_duration": dur, # Use full available duration
                    "target_duration": final_segment_dur,
                    "direction": current_direction,
                    "type": segment_type,
                    "speed": clips_speed,
                    "orig_clip_id": orig_idx
                })
                scene_idx = (scene_idx + 1) % len(scenes)

        # Fill remaining time if any
        current_gen_duration = sum(s["target_duration"] for s in segments)
        if current_gen_duration < total_audio_duration:
            remaining = total_audio_duration - current_gen_duration
            if remaining > 0.1:
                 idx = scene_idx % len(scenes)
                 (start_tc, end_tc), orig_idx = scenes[idx]
                 dur = end_tc.get_seconds() - start_tc.get_seconds()

                 current_direction = video_direction
                 if video_direction == "Random":
                    current_direction = random.choice(["Forward", "Backward"])
                 
                 req_rem = remaining
                 if active_transitions and len(segments) > 0:
                     req_rem += TRANS_DUR

                 segments.append({
                     "mode": "loop",
                     "start": start_tc.get_seconds(),
                     "src_duration": dur,
                     "target_duration": remaining,
                     "direction": current_direction,
                     "type": "Outro/Fill",
                     "speed": clips_speed,
                     "orig_clip_id": orig_idx
                 })

        if not segments:
            print("No suitable scenes matched the beats.")
            return None, "No suitable scenes matched the beats."

        # Pre-assign transitions for logging and precise rendering
        planned_transitions = []
        if active_transitions and len(segments) > 1:
            planned_transitions = [random.choice(active_transitions) for _ in range(len(segments)-1)]
            # Assign to segments for logging (transition connects THIS segment to NEXT)
            for i, t in enumerate(planned_transitions):
                segments[i]['transition'] = t

        # Generate synchronization log
        sync_log = ["Beat Sync Report:", "-----------------"]
        cumulative_time = 0.0
        for i, seg in enumerate(segments):
            start_time = cumulative_time
            end_time = cumulative_time + seg["target_duration"]
            dir_arrow = "->" if seg["direction"] == "Forward" else "<-"
            seg_type = seg.get("type", "Beat")
            speed_lbl = f"{seg['speed']}x"
            orig_clip = f"Clip #{seg.get('orig_clip_id', '?')}"
            
            trans_info = ""
            if "transition" in seg:
                trans_info = f" | Trans: {seg['transition']}"
                
            sync_log.append(f"Cut {i+1} [{seg_type}]: {start_time:.2f}s - {end_time:.2f}s | {orig_clip} | {seg['mode'].title()} | {dir_arrow} {seg['direction']} | Spd: {speed_lbl}{trans_info}")
            cumulative_time = end_time
        sync_report = "\n".join(sync_log)

        tmpdir = tempfile.mkdtemp()
        clip_paths = []
        try:
            for idx, seg in enumerate(segments):
                clip_path = os.path.join(tmpdir, f"clip_{idx+1:03}.mp4")
                
                # Target output duration of this clip file
                t_dur = seg["target_duration"]
                if active_transitions and idx > 0:
                    t_dur += TRANS_DUR
                
                # Fade Logic
                vf_fade = ""
                if not active_transitions:
                    fade_dur = max(0.1, min(0.5, t_dur / 5.0))
                    fade_out_st = t_dur - fade_dur
                    if fade_out_st < 0: fade_out_st = 0
                    vf_fade = f"fade=t=in:st=0:d={fade_dur},fade=t=out:st={fade_out_st}:d={fade_dur}"
                
                # Base args using preset video params
                base_args = encoding_params["video"] + ["-an", "-threads", "4"]
                
                # Filter Chain Construction
                filter_chain = []
                
                # 1. SetPTS (Speed / Stretch)
                # Standard formula for speed change: setpts=(1/speed)*PTS
                # If "stretch" mode is active (fallback), we need to calculate specific factor
                # to force src_duration to match t_dur exactly.
                if seg["mode"] == "stretch":
                    # Force fit
                    # factor = target / source
                    # But SetPTS multiplier is inverse of speed? 
                    # setpts=2.0*PTS means SLOWER (duration doubles)
                    # setpts=0.5*PTS means FASTER (duration halves)
                    # Factor = TargetDuration / SourceDuration
                    factor = t_dur / seg["src_duration"]
                    filter_chain.append(f"setpts={factor}*PTS")
                else:
                    # Apply user defined speed
                    # If user wants 2.0x speed (fast), multiplier is 0.5
                    pts_mult = 1.0 / seg["speed"]
                    filter_chain.append(f"setpts={pts_mult}*PTS")

                # 2. Reverse (If needed)
                if seg["direction"] == "Backward":
                    filter_chain.append("reverse")
                
                # 3. Aspect Ratio Cropping (High Quality Scaling)
                if aspect_ratio == "3:4 (TikTok/Shorts)":
                    # Target 1080x1440 (Scale height to 1440, crop center 1080)
                    filter_chain.append("scale=-1:1440:flags=lanczos,crop=1080:1440:(iw-1080)/2:0")
                elif aspect_ratio == "9:16 (Vertical)":
                    # Target 1080x1920
                    filter_chain.append("scale=-1:1920:flags=lanczos,crop=1080:1920:(iw-1080)/2:0")
                
                # 4. Fade (Only if no xfade)
                if vf_fade:
                    filter_chain.append(vf_fade.lstrip(','))
                
                # 4. Preset Filters (e.g., Scale)
                if encoding_params.get("filters"):
                    filter_chain.extend(encoding_params["filters"])
                
                # Force constant frame rate for individual clips too
                filter_chain.append("fps=30")
                
                vf_string = ",".join(filter_chain) if filter_chain else "null"

                # Single-step FFmpeg command
                input_args = []
                # Use 2-step for loop/backward stability
                if seg["mode"] == "loop" or seg["direction"] == "Backward":
                    raw_clip = os.path.join(tmpdir, f"raw_{idx}.mp4")
                    cmd_extract = [
                        "ffmpeg", "-y",
                        "-ss", f"{seg['start']:.3f}",
                        "-t", f"{seg['src_duration']:.6f}",
                        "-i", video_path,
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-an", "-threads", "4",
                        raw_clip
                    ]
                    subprocess.run(cmd_extract, check=True, capture_output=True)
                    
                    input_args = ["-i", raw_clip]
                    if seg["mode"] == "loop":
                        input_args = ["-stream_loop", "-1"] + input_args

                    cmd_proc = [
                        "ffmpeg", "-y"
                    ] + input_args + [
                        "-vf", vf_string,
                        "-t", f"{t_dur:.6f}"
                    ] + base_args + [clip_path]
                    subprocess.run(cmd_proc, check=True, capture_output=True)
                    
                else:
                    cmd_proc = [
                        "ffmpeg", "-y",
                        "-ss", f"{seg['start']:.3f}",
                        "-t", f"{seg['src_duration']:.6f}",
                        "-i", video_path,
                        "-vf", vf_string
                    ] + base_args + [
                        "-t", f"{t_dur:.6f}", 
                        clip_path
                    ]
                    subprocess.run(cmd_proc, check=True, capture_output=True)
                
                clip_paths.append(clip_path)

            output_video = "generated_amv.mp4"
            
            if active_transitions and len(clip_paths) > 1:
                # Batch Processing
                print(f"Applying transitions (Batch Processing)")
                
                BATCH_SIZE = 4
                current_clips = clip_paths
                current_transitions = list(planned_transitions) # Copy
                iteration = 0
                
                while len(current_clips) > 1:
                    next_clips = []
                    next_transitions = []
                    # Calculate number of batches
                    num_batches = math.ceil(len(current_clips) / BATCH_SIZE)
                    
                    if num_batches == 1:
                        print("Final pass rendering...")
                        render_batch(
                            current_clips, 
                            output_video, 
                            active_transitions, 
                            TRANS_DUR, 
                            encoding_params,
                            audio_path=audio_path,
                            specific_transitions=current_transitions
                        )
                        break
                    
                    print(f"Batch iteration {iteration}: Processing {num_batches} batches...")
                    
                    trans_idx_offset = 0
                    
                    for b in range(num_batches):
                        chunk = current_clips[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
                        
                        if len(chunk) == 1:
                            next_clips.append(chunk[0])
                            
                            if b < num_batches - 1:
                                gap_idx = trans_idx_offset + len(chunk) - 1
                                if gap_idx < len(current_transitions):
                                    next_transitions.append(current_transitions[gap_idx])
                            
                            trans_idx_offset += len(chunk)
                            continue
                            
                        part_name = os.path.join(tmpdir, f"iter_{iteration}_batch_{b}.mp4")
                        
                        # Extract transitions for this chunk
                        chunk_trans = []
                        if current_transitions:
                            start = trans_idx_offset
                            end = trans_idx_offset + len(chunk) - 1
                            chunk_trans = current_transitions[start:end]
                        
                        render_batch(
                            chunk, 
                            part_name, 
                            active_transitions, 
                            TRANS_DUR, 
                            encoding_params,
                            audio_path=None,
                            specific_transitions=chunk_trans
                        )
                        next_clips.append(part_name)
                        
                        # Handle gap transition
                        if b < num_batches - 1:
                            gap_idx = trans_idx_offset + len(chunk) - 1
                            if gap_idx < len(current_transitions):
                                next_transitions.append(current_transitions[gap_idx])
                        
                        trans_idx_offset += len(chunk)
                    
                    current_clips = next_clips
                    current_transitions = next_transitions
                    iteration += 1
                
            else:
                # Standard simple concatenation
                concat_list = os.path.join(tmpdir, "concat.txt")
                with open(concat_list, "w", encoding="utf-8") as f:
                    for path in clip_paths:
                        normalized_path = path.replace("\\\\", "/")
                        f.write(f"file '{normalized_path}'\n")

                concat_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list,
                    "-i", audio_path,
                    "-map", "0:v",
                    "-map", "1:a"
                ]
                
                if "video" in encoding_params:
                     concat_cmd.extend(encoding_params["video"])
                else:
                     concat_cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"])
                     
                if "audio" in encoding_params:
                     concat_cmd.extend(encoding_params["audio"])
                else:
                     concat_cmd.extend(["-c:a", "aac", "-b:a", "192k"])
                
                concat_cmd.append("-shortest")
                concat_cmd.append(output_video)
                
                subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
            print("Successfully created AMV.")
            return output_video, sync_report
        except subprocess.CalledProcessError as e:
            print("Error creating AMV with ffmpeg.")
            print("Stderr:", e.stderr)
            return None, f"Error: {str(e)}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if temp_audio_file and os.path.exists(temp_audio_file):
             os.remove(temp_audio_file)
    audio_path = original_audio_path

    # Handle Audio Cropping
    temp_audio_file = None
    if crop_start > 0 or (crop_end > 0 and crop_end > crop_start):
        print(f"Cropping audio: {crop_start}s to {crop_end if crop_end > 0 else 'End'}s")
        temp_audio_fd, temp_audio_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_audio_fd)
        temp_audio_file = temp_audio_path
        
        cmd_crop = ["ffmpeg", "-y", "-i", original_audio_path, "-ss", str(crop_start)]
        if crop_end > 0:
            cmd_crop.extend(["-to", str(crop_end)])
        cmd_crop.extend(["-c:a", "pcm_s16le", temp_audio_path]) # Convert to wav for consistency
        
        try:
            subprocess.run(cmd_crop, check=True, capture_output=True)
            audio_path = temp_audio_path
        except subprocess.CalledProcessError as e:
            print(f"Error cropping audio: {e}")
            return None, f"Error cropping audio: {str(e)}"

    print(f"Video path: {video_path}")
    print(f"Audio path: {audio_path}")

    try:
        total_audio_duration = librosa.get_duration(path=audio_path)
        last_audio_duration = total_audio_duration

        # Filter scenes based on selection
        if selected_clips is not None:
             if len(selected_clips) == 0:
                 scenes = [] # Explicitly empty if user deselected all
             else:
                 indices = []
                 for s in selected_clips:
                     try:
                         # Parse "Clip X (...)" to get X
                         # Expecting "Clip 1 (00:00 - 00:05)"
                         idx_str = s.split(" ")[1]
                         idx = int(idx_str) - 1
                         indices.append(idx)
                     except:
                         pass
                 scenes = [last_scene_list[i] for i in indices if 0 <= i < len(last_scene_list)]
        else:
            scenes = list(last_scene_list)
            
        if not scenes:
            print("No scenes selected.")
            return None, "Error: No scenes selected. Please select at least one clip."

        if shuffle_scenes:
            random.shuffle(scenes)

        segments = []
        scene_idx = 0

        # Group beats based on cut_intensity
        beat_groups = []
        
        # Handle Intro (time before first beat)
        first_beat_time = last_beat_times[0]
        has_intro = False
        if first_beat_time > 0.1:
            beat_groups.append({"duration": first_beat_time, "type": "Intro"})
            has_intro = True

        current_beat_idx = 0
        while current_beat_idx < len(last_beat_times) - 1:
            next_idx = min(current_beat_idx + int(cut_intensity), len(last_beat_times) - 1)
            start_time = last_beat_times[current_beat_idx]
            end_time = last_beat_times[next_idx]
            beat_groups.append({"duration": end_time - start_time, "type": "Beat"})
            current_beat_idx = next_idx

        for group_idx, group in enumerate(beat_groups):
            target_dur = float(group["duration"])
            segment_type = group["type"]
            
            if target_dur < 0.1:
                continue
                
            best_match = None
            search_attempts = 0
            
            # Target output duration for this segment (what we want in the AMV)
            final_segment_dur = target_dur
            
            # Adjust required source duration based on speed
            # If speed is 2.0x, we need 2x the source duration to fill the same time gap?
            # No, if we want 2 seconds of output at 2x speed, we need 4 seconds of source.
            # If we want 2 seconds of output at 0.5x speed, we need 1 second of source.
            required_src_dur_for_beat = final_segment_dur * clips_speed
            
            # Add transition padding to the *output* duration we are aiming for
            # But wait, padding needs to be available in the source too.
            # Actually, we extract `required_src` and then speed-change it to `final_segment_dur`.
            # If we have transitions, the *output* clip needs to be longer by TRANS_DUR.
            # So we need (TRANS_DUR * clips_speed) more source time.
            
            output_dur_needed = final_segment_dur
            if active_transitions and group_idx > 0:
                 output_dur_needed += TRANS_DUR
                 
            required_src_dur = output_dur_needed * clips_speed

            while search_attempts < len(scenes):
                idx = (scene_idx + search_attempts) % len(scenes)
                start_tc, end_tc = scenes[idx]
                dur = end_tc.get_seconds() - start_tc.get_seconds()
                if dur >= required_src_dur:
                    best_match = (idx, start_tc.get_seconds(), dur)
                    break
                search_attempts += 1
                
            # Determine direction for this segment
            current_direction = video_direction
            if video_direction == "Random":
                current_direction = random.choice(["Forward", "Backward"])

            if best_match:
                idx, start, dur = best_match
                segments.append({
                    "mode": "normal",
                    "start": start,
                    "src_duration": required_src_dur, # Extract exactly what we need
                    "target_duration": final_segment_dur,
                    "direction": current_direction,
                    "type": segment_type,
                    "speed": clips_speed
                })
                scene_idx = (idx + 1) % len(scenes)
            else:
                # Fallback: Stretch or Loop (Using longest clip)
                longest_idx = 0
                max_d = 0
                for i_s, (st_tc, en_tc) in enumerate(scenes):
                    d_s = en_tc.get_seconds() - st_tc.get_seconds()
                    if d_s > max_d:
                        max_d = d_s
                        longest_idx = i_s
                
                idx = longest_idx
                start_tc, end_tc = scenes[idx]
                dur = end_tc.get_seconds() - start_tc.get_seconds()
                dur = max(dur, 0.1)
                
                # If standard stretch is allowed (to fit beat)
                # If clips_speed is 1.0, normal stretch logic applies.
                # If clips_speed is != 1.0, we are already applying a speed effect.
                # In fallback, we just want to fill the gap.
                
                mode = "loop"
                # If we have enough source to cover at least half the requirement, stretch it
                if allow_stretch and dur >= required_src_dur * 0.5:
                    mode = "stretch"
                
                segments.append({
                    "mode": mode,
                    "start": start_tc.get_seconds(),
                    "src_duration": dur, # Use full available duration
                    "target_duration": final_segment_dur,
                    "direction": current_direction,
                    "type": segment_type,
                    "speed": clips_speed
                })
                scene_idx = (scene_idx + 1) % len(scenes)

        # Fill remaining time if any
        current_gen_duration = sum(s["target_duration"] for s in segments)
        if current_gen_duration < total_audio_duration:
            remaining = total_audio_duration - current_gen_duration
            if remaining > 0.1:
                 idx = scene_idx % len(scenes)
                 start_tc, end_tc = scenes[idx]
                 dur = end_tc.get_seconds() - start_tc.get_seconds()

                 current_direction = video_direction
                 if video_direction == "Random":
                    current_direction = random.choice(["Forward", "Backward"])
                 
                 req_rem = remaining
                 if active_transitions and len(segments) > 0:
                     req_rem += TRANS_DUR

                 segments.append({
                     "mode": "loop",
                     "start": start_tc.get_seconds(),
                     "src_duration": dur,
                     "target_duration": remaining,
                     "direction": current_direction,
                     "type": "Outro/Fill",
                     "speed": clips_speed
                 })

        if not segments:
            print("No suitable scenes matched the beats.")
            return None, "No suitable scenes matched the beats."

        # Generate synchronization log
        sync_log = ["Beat Sync Report:", "-----------------"]
        cumulative_time = 0.0
        for i, seg in enumerate(segments):
            start_time = cumulative_time
            end_time = cumulative_time + seg["target_duration"]
            dir_arrow = "->" if seg["direction"] == "Forward" else "<-"
            seg_type = seg.get("type", "Beat")
            speed_lbl = f"{seg['speed']}x"
            sync_log.append(f"Cut {i+1} [{seg_type}]: {start_time:.2f}s - {end_time:.2f}s | {seg['mode'].title()} | {dir_arrow} {seg['direction']} | Spd: {speed_lbl}")
            cumulative_time = end_time
        sync_report = "\n".join(sync_log)

        tmpdir = tempfile.mkdtemp()
        clip_paths = []
        try:
            for idx, seg in enumerate(segments):
                clip_path = os.path.join(tmpdir, f"clip_{idx+1:03}.mp4")
                
                # Target output duration of this clip file
                t_dur = seg["target_duration"]
                if active_transitions and idx > 0:
                    t_dur += TRANS_DUR
                
                # Fade Logic
                vf_fade = ""
                if not active_transitions:
                    fade_dur = max(0.1, min(0.5, t_dur / 5.0))
                    fade_out_st = t_dur - fade_dur
                    if fade_out_st < 0: fade_out_st = 0
                    vf_fade = f"fade=t=in:st=0:d={fade_dur},fade=t=out:st={fade_out_st}:d={fade_dur}"
                
                # Base args using preset video params
                base_args = encoding_params["video"] + ["-an", "-threads", "4"]
                
                # Filter Chain Construction
                filter_chain = []
                
                # 1. SetPTS (Speed / Stretch)
                # Standard formula for speed change: setpts=(1/speed)*PTS
                # If "stretch" mode is active (fallback), we need to calculate specific factor
                # to force src_duration to match t_dur exactly.
                if seg["mode"] == "stretch":
                    # Force fit
                    # factor = target / source
                    # But SetPTS multiplier is inverse of speed? 
                    # setpts=2.0*PTS means SLOWER (duration doubles)
                    # setpts=0.5*PTS means FASTER (duration halves)
                    # Factor = TargetDuration / SourceDuration
                    factor = t_dur / seg["src_duration"]
                    filter_chain.append(f"setpts={factor}*PTS")
                else:
                    # Apply user defined speed
                    # If user wants 2.0x speed (fast), multiplier is 0.5
                    pts_mult = 1.0 / seg["speed"]
                    filter_chain.append(f"setpts={pts_mult}*PTS")

                # 2. Reverse (If needed)
                if seg["direction"] == "Backward":
                    filter_chain.append("reverse")
                
                # 3. Aspect Ratio Cropping (High Quality Scaling)
                if aspect_ratio == "3:4 (TikTok/Shorts)":
                    # Target 1080x1440 (Scale height to 1440, crop center 1080)
                    filter_chain.append("scale=-1:1440:flags=lanczos,crop=1080:1440:(iw-1080)/2:0")
                elif aspect_ratio == "9:16 (Vertical)":
                    # Target 1080x1920
                    filter_chain.append("scale=-1:1920:flags=lanczos,crop=1080:1920:(iw-1080)/2:0")
                
                # 4. Fade (Only if no xfade)
                if vf_fade:
                    filter_chain.append(vf_fade.lstrip(','))
                
                # 4. Preset Filters (e.g., Scale)
                if encoding_params.get("filters"):
                    filter_chain.extend(encoding_params["filters"])
                
                # Force constant frame rate for individual clips too
                filter_chain.append("fps=30")
                
                vf_string = ",".join(filter_chain) if filter_chain else "null"

                # Single-step FFmpeg command
                input_args = []
                # Use 2-step for loop/backward stability
                if seg["mode"] == "loop" or seg["direction"] == "Backward":
                    raw_clip = os.path.join(tmpdir, f"raw_{idx}.mp4")
                    cmd_extract = [
                        "ffmpeg", "-y",
                        "-ss", f"{seg['start']:.3f}",
                        "-t", f"{seg['src_duration']:.6f}",
                        "-i", video_path,
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-an", "-threads", "4",
                        raw_clip
                    ]
                    subprocess.run(cmd_extract, check=True, capture_output=True)
                    
                    input_args = ["-i", raw_clip]
                    if seg["mode"] == "loop":
                        input_args = ["-stream_loop", "-1"] + input_args

                    cmd_proc = [
                        "ffmpeg", "-y"
                    ] + input_args + [
                        "-vf", vf_string,
                        "-t", f"{t_dur:.6f}"
                    ] + base_args + [clip_path]
                    subprocess.run(cmd_proc, check=True, capture_output=True)
                    
                else:
                    cmd_proc = [
                        "ffmpeg", "-y",
                        "-ss", f"{seg['start']:.3f}",
                        "-t", f"{seg['src_duration']:.6f}",
                        "-i", video_path,
                        "-vf", vf_string
                    ] + base_args + [
                        "-t", f"{t_dur:.6f}", 
                        clip_path
                    ]
                    subprocess.run(cmd_proc, check=True, capture_output=True)
                
                clip_paths.append(clip_path)

            output_video = "generated_amv.mp4"
            
            if active_transitions and len(clip_paths) > 1:
                # Batch Processing
                print(f"Applying transitions from: {selected_transitions} (Batch Processing)")
                
                BATCH_SIZE = 4
                current_clips = clip_paths
                iteration = 0
                
                while len(current_clips) > 1:
                    next_clips = []
                    # Calculate number of batches
                    num_batches = math.ceil(len(current_clips) / BATCH_SIZE)
                    
                    if num_batches == 1:
                        print("Final pass rendering...")
                        render_batch(
                            current_clips, 
                            output_video, 
                            active_transitions, 
                            TRANS_DUR, 
                            encoding_params,
                            audio_path=audio_path
                        )
                        break
                    
                    print(f"Batch iteration {iteration}: Processing {num_batches} batches...")
                    for b in range(num_batches):
                        chunk = current_clips[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
                        
                        if len(chunk) == 1:
                            next_clips.append(chunk[0])
                            continue
                            
                        part_name = os.path.join(tmpdir, f"iter_{iteration}_batch_{b}.mp4")
                        render_batch(
                            chunk, 
                            part_name, 
                            active_transitions, 
                            TRANS_DUR, 
                            encoding_params,
                            audio_path=None
                        )
                        next_clips.append(part_name)
                    
                    current_clips = next_clips
                    iteration += 1
                
            else:
                # Standard simple concatenation
                concat_list = os.path.join(tmpdir, "concat.txt")
                with open(concat_list, "w", encoding="utf-8") as f:
                    for path in clip_paths:
                        normalized_path = path.replace("\\\\", "/")
                        f.write(f"file '{normalized_path}'\n")

                concat_cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_list,
                    "-i", audio_path,
                    "-map", "0:v",
                    "-map", "1:a"
                ]
                
                if "video" in encoding_params:
                     concat_cmd.extend(encoding_params["video"])
                else:
                     concat_cmd.extend(["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20"])
                     
                if "audio" in encoding_params:
                     concat_cmd.extend(encoding_params["audio"])
                else:
                     concat_cmd.extend(["-c:a", "aac", "-b:a", "192k"])
                
                concat_cmd.append("-shortest")
                concat_cmd.append(output_video)
                
                subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
            print("Successfully created AMV.")
            return output_video, sync_report
        except subprocess.CalledProcessError as e:
            print("Error creating AMV with ffmpeg.")
            print("Stderr:", e.stderr)
            return None, f"Error: {str(e)}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
        if temp_audio_file and os.path.exists(temp_audio_file):
             os.remove(temp_audio_file)

with gr.Blocks() as demo:
    gr.Markdown("## AutoAMV")
    gr.Markdown("Upload a video and a song to create a synchronized montage.")
    
    with gr.Tabs():
        with gr.TabItem("Song"):
            with gr.Column():
                audio_input = gr.File(label="Song", file_types=["audio"])
                
                with gr.Accordion("Rhythm Detection Settings", open=True):
                    algo_dropdown = gr.Dropdown(
                        label="Detection Algorithm",
                        choices=["Standard", "Low Percussion"],
                        value="Standard",
                        info="Standard: For drums/beats | Low Percussion: For piano/strings/ambient (uses pitch detection)"
                    )
                    tightness_slider = gr.Slider(
                        minimum=1, 
                        maximum=200, 
                        value=100, 
                        step=1, 
                        label="Tightness (Rigidez)", 
                        info="100 = Standard (Batería constante) | 30 = Flexible (Música expresiva/Rubato)"
                    )
                    with gr.Row():
                        crop_start = gr.Number(label="Start Time (s) - e.g. 12.50", value=0, precision=None)
                        crop_end = gr.Number(label="End Time (s) - e.g. 45.30", value=0, precision=None, info="0 = End of file")
                    
                    with gr.Row():
                        preview_btn = gr.Button("Preview Cut")
                        preview_output = gr.Audio(label="Cut Preview", type="filepath")
                    
                    preview_btn.click(
                        fn=preview_audio_cut,
                        inputs=[audio_input, crop_start, crop_end],
                        outputs=preview_output
                    )

                rhythm_button = gr.Button("Detect Rhythm")
                rhythm_output = gr.Plot(label="Rhythm Visualization")
                info_output = gr.Textbox(label="Beat Info", lines=20, max_lines=50, interactive=False)
        with gr.TabItem("Clips"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.File(label="Video", file_types=["video"])
                    
                    with gr.Accordion("Scene Detection Settings", open=True):
                        detector_type = gr.Dropdown(
                            label="Detector Type",
                            choices=["detect-content", "detect-threshold", "detect-adaptive"],
                            value="detect-content",
                            info="Content: Standard | Threshold: Fades/Flashes | Adaptive: Moving cam/Grain"
                        )
                        threshold_slider = gr.Slider(
                            minimum=0, maximum=100, step=0.5, value=30.0, 
                            label="Threshold (Umbral de Cambio)",
                            info="Menor valor = Menos cortes. Mayor valor = Más cortes."
                        )
                        min_scene_len = gr.Number(
                            label="Min Scene Length (seconds)", 
                            value=0.6, 
                            step=0.1,
                            info="Minimum duration of a scene to be accepted."
                        )
                        frame_window = gr.Number(
                            label="Frame Window (Fotogramas Ventana)",
                            value=2,
                            step=1,
                            info="Cuántos fotogramas hacia atrás y adelante se usan para calcular el promedio de movimiento.",
                            visible=False # Hidden by default
                        )
                        min_content_val = gr.Number(
                            label="Min Content Value (Valor Mínimo Contenido)",
                            value=15.0,
                            step=0.1,
                            info="Cambio absoluto mínimo que debe ocurrir, independientemente del promedio.",
                            visible=False # Hidden by default
                        )
                        
                        preview_mode = gr.Dropdown(
                            label="Preview Mode",
                            choices=["Thumbnails", "Video (Fast/Copy)", "Video (High Quality)"],
                            value="Thumbnails",
                            info="Thumbnails: Images only. Video: Splits clips for playback (Fast=Copy, HQ=Re-encode)."
                        )
                    
                    clips_button = gr.Button("Generate Clips", variant="primary")
                    
                    detector_type.change(
                        fn=lambda x: [
                            gr.update(label="Threshold (Umbral de Brillo)", value=12.0, minimum=0, maximum=255, info="Umbral para detectar negro. Menor = Estricto (solo negro puro). Mayor = Acepta gris oscuro.") if x == "detect-threshold" 
                            else (gr.update(label="Threshold (Umbral Adaptativo)", value=3.0, minimum=1.0, maximum=20.0, info="Define cuánto más alto debe ser el cambio actual comparado con el promedio de fotogramas anteriores para considerarse un corte. Menor = Más sensible, Mayor = Más estricto.") if x == "detect-adaptive" 
                            else gr.update(label="Threshold (Umbral de Cambio)", value=30.0, minimum=0, maximum=100, info="Menor valor = Menos cortes. Mayor valor = Más cortes.")),
                            gr.update(visible=x == "detect-adaptive"), # frame_window visibility
                            gr.update(visible=x == "detect-adaptive")  # min_content_val visibility
                        ], 
                        inputs=detector_type, 
                        outputs=[threshold_slider, frame_window, min_content_val]
                    )

                with gr.Column(scale=2):
                    scenes_gallery = gr.Gallery(label="Scene Previews", show_label=True, elem_id="gallery", columns=4, height="auto")
                    with gr.Row():
                        select_all_scenes = gr.Button("Select All Scenes", size="sm")
                        deselect_all_scenes = gr.Button("Deselect All Scenes", size="sm")
                    scenes_selector = gr.CheckboxGroup(
                        label="Select Scenes to Use", 
                        choices=[], 
                        value=[], 
                        info="Uncheck scenes you want to exclude."
                    )
                
                select_all_scenes.click(fn=lambda: gr.update(value=[f"Clip {i+1} ({last_scene_list[i][0].get_timecode()} - {last_scene_list[i][1].get_timecode()}) [{last_scene_list[i][1].get_seconds()-last_scene_list[i][0].get_seconds():.2f}s]" for i in range(len(last_scene_list))]), outputs=scenes_selector)
                deselect_all_scenes.click(fn=lambda: gr.update(value=[]), outputs=scenes_selector)

        with gr.TabItem("Generated AMV"):
            with gr.Row():
                with gr.Column():
                    cut_intensity = gr.Slider(minimum=1, maximum=16, step=1, value=1, label="Cut Intensity (Beats per Cut)")
                    clips_speed = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Clips Speed (Multiplier)", info="0.5x = Slow Motion, 2.0x = Fast Forward")
                    video_direction = gr.Radio(["Forward", "Backward", "Random"], value="Forward", label="Video Direction")
                    quality_preset = gr.Dropdown(
                        label="Quality Preset",
                        choices=list(PRESETS.keys()),
                        value="Default (CPU Balanced)",
                        info="Select encoding quality and hardware acceleration."
                    )
                    aspect_ratio_dropdown = gr.Dropdown(
                        label="Aspect Ratio",
                        choices=["Original (16:9)", "3:4 (TikTok/Shorts)", "9:16 (Vertical)"],
                        value="Original (16:9)",
                        info="Choose output aspect ratio. Center crop will be applied for vertical formats."
                    )
                    transition_choices = [
                        "Flash Blanco (Fade White)", 
                        "Zoom In (Zoom In)", 
                        "Radar (Radial)",
                        "Deslizar Izq (Slide Left)",
                        "Deslizar Der (Slide Right)",
                        "Barrido (Wipe Right)",
                        "Elástico (Smooth Left)",
                        "Pixelar (Pixelize)",
                        "Desenfoque (H-Blur)",
                        "Aplastar (Squeeze V)",
                        "Círculo (Circle Open)"
                    ]
                    with gr.Accordion("Transition Effects", open=False):
                        with gr.Row():
                            select_all_trans = gr.Button("Select All", size="sm")
                            deselect_all_trans = gr.Button("Deselect All", size="sm")
                        selected_transitions = gr.CheckboxGroup(
                            label="Effects",
                            choices=transition_choices,
                            value=[],
                            info="Select multiple to apply randomly. If none selected, uses simple cut.",
                            show_label=False
                        )
                    
                    select_all_trans.click(fn=lambda: transition_choices, outputs=selected_transitions)
                    deselect_all_trans.click(fn=lambda: [], outputs=selected_transitions)
                    shuffle_scenes = gr.Checkbox(label="Shuffle Scenes (Randomize Order)", value=False)
                    allow_stretch = gr.Checkbox(label="Allow Stretching (Estirar clips)", value=True)
                    create_button = gr.Button("Create AMV")
                with gr.Column():
                    video_output = gr.Video(label="Generated AMV", format="mp4")
                    sync_output = gr.Textbox(label="Synchronization Log", lines=10)
    
        with gr.TabItem("Filters"):
            gr.Markdown("### Video Enhancer (Filters)")
            with gr.Row():
                with gr.Column():
                    en_video = gr.File(label="Video to Enhance", file_types=["video"])
                    en_preset = gr.Dropdown(
                        label="Filter Preset",
                        choices=["CLEAN (Natural/Cine)", "VIBRANT (Anime/Standard)", "AGGRESSIVE (Max/Epic)", "CUSTOM"],
                        value="VIBRANT (Anime/Standard)",
                        info="Choose a preset style."
                    )
                    with gr.Group(visible=False) as custom_group:
                        en_cas = gr.Slider(0.0, 1.0, value=0.6, label="CAS (Sharpness)", step=0.1)
                        en_contrast = gr.Slider(0.5, 2.0, value=1.0, label="Contrast", step=0.05)
                        en_saturation = gr.Slider(0.0, 3.0, value=1.0, label="Saturation", step=0.1)
                        en_brightness = gr.Slider(-1.0, 1.0, value=0.0, label="Brightness", step=0.01)
                        en_gamma = gr.Slider(0.1, 5.0, value=1.0, label="Gamma", step=0.1)
                    
                    en_btn = gr.Button("Apply Filter", variant="primary")
                
                with gr.Column():
                    en_output = gr.Video(label="Enhanced Video")
                    en_log = gr.Textbox(label="Log")

            en_preset.change(
                fn=lambda x: gr.update(visible=x == "CUSTOM"),
                inputs=en_preset,
                outputs=custom_group
            )
            
            en_btn.click(
                fn=apply_video_filter,
                inputs=[en_video, en_preset, en_cas, en_contrast, en_saturation, en_brightness, en_gamma],
                outputs=[en_output, en_log]
            )

        with gr.TabItem("Subtitle"):
            gr.Markdown("### Subtitle Burner (Karaoke Effect)")
            with gr.Row():
                with gr.Column():
                    ex_video = gr.File(label="Video to Subtitle", file_types=["video"])
                    ex_srt = gr.File(label="Subtitle File (.srt)", file_types=[".srt"])
                    ex_btn = gr.Button("Burn Subtitles", variant="primary")
                with gr.Column():
                    ex_output = gr.Video(label="Video con Subtítulos")
                    ex_log = gr.Textbox(label="Log")
                    ex_ass_output = gr.File(label="Generated ASS File")
            
            ex_btn.click(
                fn=burn_subtitles,
                inputs=[ex_video, ex_srt],
                outputs=[ex_output, ex_log, ex_ass_output]
            )
    
    create_button.click(
        fn=create_amv,
        inputs=[video_input, audio_input, cut_intensity, video_direction, shuffle_scenes, allow_stretch, selected_transitions, quality_preset, clips_speed, crop_start, crop_end, aspect_ratio_dropdown, scenes_selector],
        outputs=[video_output, sync_output],
    )
    
    rhythm_button.click(
        fn=detect_rhythm,
        inputs=[audio_input, algo_dropdown, tightness_slider, crop_start, crop_end],
        outputs=[rhythm_output, info_output],
    )
    
    clips_button.click(
        fn=detect_scenes,
        inputs=[video_input, detector_type, threshold_slider, min_scene_len, frame_window, min_content_val, preview_mode],
        outputs=[scenes_gallery, scenes_selector],
    )

if __name__ == "__main__":
    demo.launch()