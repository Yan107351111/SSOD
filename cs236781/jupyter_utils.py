import base64
import os
import pathlib
import subprocess
import time

import IPython.display


def nbconvert(nb_path, execute=False, inplace=False, clear_output=False,
              debug=False, stdout=False, allow_errors=False, timeout_sec=3600):
    args = ["jupyter", "nbconvert"]
    if execute:
        args.append("--execute")
    if allow_errors:
        args.append("--allow-errors")
    if clear_output:
        args.append("--ClearOutputPreprocessor.enabled=True")
    if inplace or clear_output:
        args.append("--inplace")
    if debug:
        args.append("--debug")
    if stdout:
        args.append("--stdout")
    if timeout_sec is not None:
        args.append(f"--ExecutePreprocessor.timeout={timeout_sec}")
    args.append(nb_path)

    true_flags = []
    for k, v in locals().items():
        if v is True:
            true_flags.append(k)
    true_flags = str.join('|', true_flags)

    print(f'>> Running nbconvert on notebook {nb_path} [{true_flags}]')
    ts = time.time()
    subprocess.check_output(args)
    print(f'>> Finished nbconvert on notebook {nb_path}, '
          f'elapsed={time.time() - ts:.3f}s')


def nbmerge(nb_paths, output_filename):
    if not output_filename.endswith('.ipynb'):
        output_filename += '.ipynb'

    args = ['nbmerge', '-o', output_filename, '-v']
    args.extend(nb_paths)

    nb_names = [pathlib.Path(nb_path).stem for nb_path in nb_paths]
    print(f'>> Running nbmerge on notebooks {str.join(", ", nb_names)}')

    subprocess.check_output(args)


def show_video_in_notebook(video_path, width=500, height='auto', autoplay=True,
                           embed=True):
    """
    Helper function to show a video in a jupyter notebook.
    :param video_path: Path to video file.
    :param width: Width of video element on the page.
    :param height: Height of video element on the page.
    :param autoplay: Whether video should automatically start playing.
    :param embed: Whether to embed the video in the notebook itself,
    or just link to it. Linking won't work if file is outside servers pwd;
    embedding won't work if video is too large.
    :return: An IPython HTML object that jupyter notebook can display.
    """
    video_path = os.path.relpath(video_path, start=os.path.curdir)
    autoplay = 'autoplay' if autoplay else ''

    if embed:
        with open(video_path, "rb") as f:
            encoded = base64.b64encode(f.read(), ).decode("ascii")
        _, ext = os.path.splitext(video_path)
        src_str = f"data:video/{ext[1:]};base64,{encoded}"
    else:
        src_str = f"{video_path}"

    raw_html = f'<video src="{src_str}" controls {autoplay} ' \
               f'width="{width}" height="{height}" />'

    return IPython.display.HTML(data=raw_html)


