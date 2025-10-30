import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+M]))
    
    if M > N:
        print("No")
        return
    
    # Compute leftmost positions
    left_pos = []
    j = 0
    for i in range(N):
        if j < M and A[i] == B[j]:
            left_pos.append(i)
            j += 1
    if j < M:
        print("No")
        return
    
    # Compute rightmost positions
    right_pos = [0] * M
    j = M - 1
    for i in reversed(range(N)):
        if j >= 0 and A[i] == B[j]:
            right_pos[j] = i
            j -= 1
    if j >= 0:
        print("No")
        return
    
    if left_pos != right_pos:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()