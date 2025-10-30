# YOUR CODE HERE
def count_matching_lengths(strings):
    count = 0
    for i in range(1, 13):
        if len(strings[i - 1]) == i:
            count += 1
    return count

if __name__ == "__main__":
    import sys
    input_strings = [sys.stdin.readline().strip() for _ in range(12)]
    result = count_matching_lengths(input_strings)
    print(result)