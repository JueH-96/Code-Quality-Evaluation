def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    chords = []
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        index += 2
        start = min(A, B)
        end = max(A, B)
        chords.append((start, end))
    
    # Sort chords based on starting point
    chords_sorted = sorted(chords, key=lambda x: x[0])
    
    # Check if endpoints are in decreasing order
    prev_end = chords_sorted[0][1]
    for i in range(1, N):
        current_end = chords_sorted[i][1]
        if current_end > prev_end:
            print("Yes")
            return
        prev_end = current_end
    print("No")

if __name__ == "__main__":
    main()