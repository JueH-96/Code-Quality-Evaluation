import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    nums = list(map(int, data[1:1 + n]))

    last_pos = dict()           # value -> last index (1-based)
    res = []

    for idx, val in enumerate(nums, 1):   # idx is 1-based
        prev = last_pos.get(val, -1)
        res.append(str(prev))
        last_pos[val] = idx

    sys.stdout.write(" ".join(res))

if __name__ == "__main__":
    main()