import subprocess


def tsp_submit(*cmds):

    tsp_cmd = ["tsp"]
    tsp_cmd.extend(cmds)
    result = subprocess.run(tsp_cmd, capture_output=True, text=True)
    return int(result.stdout.strip())


def _form_row_header(N):

    row_0 = ["|"] * (N + 1)
    row_0 = (" " * 3).join(row_0)
    row_1 = ["|"] * (N + 1)
    row_1 = " --- ".join(row_1)

    return "\n".join([row_0, row_1])


def form_in_md_rows(*datas):

    N = len(datas[0])
    for data in datas:
        assert N == len(data)

    header = _form_row_header(N)

    prefix = "| "
    suffix = " |"

    rows = []
    for data in datas:
        row = map(str, data)
        row = " | ".join(row)
        row = prefix + row + suffix
        rows.append(row)

    return "\n".join([header] + rows)
