def can_make_non_decreasing(N, A):
    total = 0
    for i in range(N):
        total += A[i]
        # Check if the total so far is less than the required minimum
        if total < i * (i + 1) // 2:
            return "No"
    return "Yes"

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    T = int(data[0])
    results = []
    index = 1
    
    for _ in range(T):
        N = int(data[index])
        A = list(map(int, data[index + 1].split()))
        index += 2
        
        result = can_make_non_decreasing(N, A)
        results.append(result)
    
    print("
".join(results))

if __name__ == "__main__":
    main()