# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    freq = [0] * (N + 2)
    for x in A:
        freq[x] +=1
    
    cnt_before = [0] * (N +2)
    cnt_after = freq.copy()
    
    total_sum = 0
    ans = 0
    
    for j in range(N):
        x = A[j]
        # Calculate contribution
        contribution = total_sum - (cnt_before[x] * cnt_after[x])
        ans += contribution
        # Update cnt_after
        cnt_after[x] -=1
        # Update total_sum by subtracting cnt_before[x]
        total_sum -= cnt_before[x]
        # Update cnt_before
        cnt_before[x] +=1
        # Update total_sum by adding cnt_before[x] * cnt_after[x]
        total_sum += cnt_before[x] * cnt_after[x]
    
    print(ans)

if __name__ == "__main__":
    main()