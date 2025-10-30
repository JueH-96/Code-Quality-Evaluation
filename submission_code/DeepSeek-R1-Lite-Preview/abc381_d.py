def main():
    import sys
    from collections import defaultdict

    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))

    freq = defaultdict(int)
    unique = 0
    count_two = 0
    left = 0
    max_length = 0

    for right in range(N):
        current_element = A[right]
        prev_freq = freq[current_element]
        freq[current_element] += 1

        if prev_freq == 0:
            unique += 1
        if freq[current_element] == 2:
            count_two += 1
        elif freq[current_element] == 3:
            while freq[current_element] > 2:
                left_element = A[left]
                freq[left_element] -= 1
                if freq[left_element] == 1:
                    count_two -= 1
                elif freq[left_element] == 0:
                    unique -= 1
                left += 1

        window_length = right - left + 1
        if unique * 2 == window_length and count_two == unique:
            max_length = max(max_length, window_length)

    print(max_length)

if __name__ == '__main__':
    main()