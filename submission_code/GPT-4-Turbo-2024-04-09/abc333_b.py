def segment_length(p1, p2):
    positions = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    idx1 = positions[p1]
    idx2 = positions[p2]
    return min(abs(idx1 - idx2), 5 - abs(idx1 - idx2))

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    S1, S2 = data[0][0], data[0][1]
    T1, T2 = data[1][0], data[1][1]
    
    if segment_length(S1, S2) == segment_length(T1, T2):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()