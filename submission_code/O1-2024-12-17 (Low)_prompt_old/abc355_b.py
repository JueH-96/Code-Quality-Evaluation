def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:2+N+M]))

    # Combine the sequences along with a marker to indicate whether an element belongs to A or B
    combined = [(val, 'A') for val in A] + [(val, 'B') for val in B]
    # Sort by the numeric value
    combined.sort(key=lambda x: x[0])

    # Check for two consecutive 'A' elements after sorting
    for i in range(len(combined) - 1):
        if combined[i][1] == 'A' and combined[i+1][1] == 'A':
            print("Yes")
            return

    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()