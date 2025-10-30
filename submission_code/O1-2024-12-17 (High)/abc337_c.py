def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Array to store who comes after i (1-based index).
    # If next_person[i] = j, it means j is immediately behind i in line.
    next_person = [0] * (N + 1)
    
    front = 0  # Will store the index of the person at the front.
    for i in range(1, N + 1):
        if A[i - 1] == -1:
            front = i
        else:
            next_person[A[i - 1]] = i

    # Traverse the line from front to back.
    order = []
    current = front
    for _ in range(N):
        order.append(current)
        current = next_person[current]

    # Print the result.
    print(" ".join(map(str, order)))

# Do not remove the function call below
main()