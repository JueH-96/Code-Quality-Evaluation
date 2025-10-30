def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]

    # dp0 = number of subarrays ending at the current position with final NAND = 0
    # dp1 = number of subarrays ending at the current position with final NAND = 1
    dp0, dp1 = 0, 0
    answer = 0

    for ch in S:
        if ch == '0':
            # A_j == 0
            new_dp0 = 1                 # Starting new subarray of length 1 gives final=0 if bit=0
            new_dp1 = dp0 + dp1        # From previous partial=0 or =1, reading 0 => final=1
        else:
            # A_j == 1
            new_dp0 = dp1             # Extending subarray with prev partial=1 and bit=1 => final=0
            new_dp1 = dp0 + 1         # Extending subarray with prev partial=0 => final=1, 
                                      # plus starting a new subarray of length1 => final=1 if bit=1
        dp0, dp1 = new_dp0, new_dp1
        answer += dp1  # We add how many subarrays ending HERE have final=1

    print(answer)

# Do not forget to call main()
main()