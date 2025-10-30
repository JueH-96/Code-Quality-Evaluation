import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    MOD = 10**9 + 7
    total_sum = 0
    current_sum = 0
    multiplier = 1
    
    # Traverse the string from the end to the start
    for i in range(N - 1, -1, -1):
        # Update the current sum for the current digit
        current_sum = (current_sum + int(S[i]) * multiplier) % MOD
        # Add the current sum to the total sum
        total_sum = (total_sum + current_sum) % MOD
        # Update the multiplier for the next digit
        multiplier = (multiplier * 10 + 1) % MOD
    
    print(total_sum)

if __name__ == "__main__":
    main()