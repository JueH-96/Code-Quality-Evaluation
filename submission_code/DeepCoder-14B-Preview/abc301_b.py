n = int(input())
A = list(map(int, input().split()))

while True:
    found = False
    for i in range(len(A) - 1):
        if abs(A[i] - A[i+1]) != 1:
            # Process this pair
            if A[i] < A[i+1]:
                current = A[i] + 1
                insert = []
                while current < A[i+1]:
                    insert.append(current)
                    current += 1
            else:
                current = A[i] - 1
                insert = []
                while current > A[i+1]:
                    insert.append(current)
                    current -= 1
            # Insert into A
            A = A[:i+1] + insert + A[i+1:]
            found = True
            break  # Process only the first found pair and restart the scan
    if not found:
        break

print(' '.join(map(str, A)))