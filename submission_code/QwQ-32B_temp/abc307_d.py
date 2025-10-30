import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()

    stack = []
    intervals = []

    for i, c in enumerate(S):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                start = stack.pop()
                intervals.append((start, i))

    if not intervals:
        print(S)
        return

    # Sort intervals by their start
    intervals.sort()

    merged = []
    current_start, current_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= current_end:
            current_end = max(current_end, e)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = s, e
    merged.append((current_start, current_end))

    result = []
    current_pos = 0
    for s, e in merged:
        if current_pos < s:
            result.append(S[current_pos:s])
        current_pos = e + 1
    result.append(S[current_pos:])

    print(''.join(result))

if __name__ == "__main__":
    main()