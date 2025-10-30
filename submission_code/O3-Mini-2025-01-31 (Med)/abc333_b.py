def chord_type(segment):
    # Map vertices to indices (A:0, B:1, C:2, D:3, E:4)
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    a, b = mapping[segment[0]], mapping[segment[1]]
    diff = abs(a - b)
    # The chord type is determined by the minimal circular distance
    return min(diff, 5 - diff)

def main():
    import sys
    # Read input from standard input
    data = sys.stdin.read().split()
    # Expect two segments: S_1S_2 and T_1T_2
    if len(data) < 2:
        return
    s_seg, t_seg = data[0], data[1]
    
    # Get the chord types, which are the minimal distances
    if chord_type(s_seg) == chord_type(t_seg):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()