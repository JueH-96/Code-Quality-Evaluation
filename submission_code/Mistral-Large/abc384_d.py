import sys

def can_form_sum(N, S, A):
    # Calculate the sum of the first N elements
    total_sum = sum(A)

    # If S is greater than the total sum of one period, it's not possible
    if S > total_sum:
        return "No"

    # If S is equal to the total sum of one period, it's possible
    if S == total_sum:
        return "Yes"

    # Calculate the prefix sums
    prefix_sums = [0] * (2 * N + 1)
    for i in range(1, 2 * N + 1):
        prefix_sums[i] = prefix_sums[i - 1] + A[(i - 1) % N]

    # Use a set to store the prefix sums we have seen
    seen_sums = set()

    # Check if there is a subsequence with sum S
    for i in range(1, 2 * N + 1):
        current_sum = prefix_sums[i]
        if current_sum == S:
            return "Yes"
        if (current_sum - S) in seen_sums:
            return "Yes"
        seen_sums.add(current_sum)

    return "No"

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    S = int(data[1])
    A = list(map(int, data[2:]))

    result = can_form_sum(N, S, A)
    print(result)

if __name__ == "__main__":
    main()