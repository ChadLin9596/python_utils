import subprocess


def tsp_submit(*cmds):

    tsp_cmd = ["tsp"]
    tsp_cmd.extend(cmds)
    result = subprocess.run(tsp_cmd, capture_output=True, text=True)
    return int(result.stdout.strip())
