def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # next_person[i] will be the person who stands immediately behind person i
    next_person = [-1] * (N + 1)  # 1-based indexing; 0th unused

    front_person = None
    for i in range(1, N + 1):
        if A[i - 1] == -1:
            front_person = i
        else:
            next_person[A[i - 1]] = i

    # Starting from the front, traverse the line
    result = []
    curr = front_person
    while curr != -1:
        result.append(curr)
        curr = next_person[curr]

    print(" ".join(map(str, result)))

def main():
    solve()

if __name__ == "__main__":
    main()