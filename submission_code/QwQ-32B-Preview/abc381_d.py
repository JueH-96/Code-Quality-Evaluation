def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    freq = {}
    left = 0
    max_length = 0
    total_unique = 0
    count_freq_two = 0
    
    for right in range(N):
        num = A[right]
        freq[num] = freq.get(num, 0) + 1
        
        if freq[num] == 2:
            count_freq_two += 1
        if freq[num] == 1:
            total_unique += 1
        
        # Ensure no number has frequency less than 2 and adjust the window
        while freq[num] < 2:
            left_num = A[left]
            freq[left_num] -= 1
            if freq[left_num] == 1:
                total_unique -= 1
            if freq[left_num] == 0:
                del freq[left_num]
            left += 1
        
        # Check if all unique numbers appear exactly twice and the length is even
        if total_unique == count_freq_two and (right - left + 1) % 2 == 0:
            max_length = max(max_length, right - left + 1)
    
    print(max_length)

if __name__ == "__main__":
    main()