import sys

def max_alternating_sum(arr):
    n = len(arr)
    if n == 0:
        return 0

    max_ending_here_odd = max_ending_here_even = -float('inf')
    max_so_far = -float('inf')

    for i in range(n):
        if i > 0 and (arr[i] % 2 == arr[i-1] % 2):
            max_ending_here_odd = max(arr[i], max_ending_here_even + arr[i])
            max_ending_here_even = max(arr[i], max_ending_here_odd + arr[i])
        else:
            max_ending_here_odd = max(arr[i], max_ending_here_odd + arr[i])
            max_ending_here_even = max(arr[i], max_ending_here_even + arr[i])

        max_so_far = max(max_so_far, max_ending_here_odd, max_ending_here_even)

    return max_so_far

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    t = int(data[index])
    index += 1
    results = []

    for _ in range(t):
        n = int(data[index])
        index += 1
        arr = list(map(int, data[index:index + n]))
        index += n
        results.append(max_alternating_sum(arr))

    sys.stdout.write("
".join(map(str, results)) + "
")

if __name__ == "__main__":
    main()