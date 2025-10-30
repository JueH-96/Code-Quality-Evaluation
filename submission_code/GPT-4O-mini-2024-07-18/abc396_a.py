# YOUR CODE HERE
def has_three_consecutive(A):
    N = len(A)
    for i in range(N - 2):
        if A[i] == A[i + 1] == A[i + 2]:
            return True
    return False

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N + 1]))
    
    if has_three_consecutive(A):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()