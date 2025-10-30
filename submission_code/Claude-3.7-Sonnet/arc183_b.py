def can_transform(N, K, A, B):
    for i in range(N):
        found = False
        for j in range(max(0, i - K), min(N, i + K + 1)):
            if A[j] == B[i]:
                found = True
                break
        if not found:
            return False
    
    return True

def main():
    T = int(input().strip())
    
    for _ in range(T):
        N, K = map(int, input().strip().split())
        A = list(map(int, input().strip().split()))
        B = list(map(int, input().strip().split()))
        
        if can_transform(N, K, A, B):
            print("Yes")
        else:
            print("No")

if __name__ == "__main__":
    main()