def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    S = data[2]

    segments = []
    current_segment = []

    for c in S:
        if c == '0':
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(c)
    if current_segment:
        segments.append(current_segment)

    max_logo = 0
    for seg in segments:
        meal = seg.count('1')
        event = seg.count('2')
        required = event + max(0, meal - M)
        if required > max_logo:
            max_logo = required

    print(max_logo)

if __name__ == "__main__":
    main()