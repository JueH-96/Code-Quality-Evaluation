import sys

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    pos = [0]*(N+1)
    for idx, val in enumerate(A):
        pos[val] = idx
    
    operations = []
    for i in range(N):
        correct_val = i+1
        if A[i] != correct_val:
            j = pos[correct_val]
            # Swap in A
            A[i], A[j] = A[j], A[i]
            # Update positions
            pos[A[j]] = j
            pos[A[i]] = i
            operations.append((i+1, j+1))
    
    print(len(operations))
    for op in operations:
        print(op[0], op[1])

if __name__ == "__main__":
    main()