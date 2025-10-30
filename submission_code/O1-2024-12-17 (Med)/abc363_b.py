def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, T, P = map(int, data[:3])
    L = list(map(int, data[3:]))

    # Count how many already have hair length >= T
    count_at_least_T = sum(1 for x in L if x >= T)

    # If we already have P or more, answer is 0
    if count_at_least_T >= P:
        print(0)
        return

    # Otherwise, find how many more are needed
    needed = P - count_at_least_T

    # For those who have hair < T, calculate how many days until they reach T
    days_needed = [T - x for x in L if x < T]
    days_needed.sort()

    # The answer is the (needed)-th smallest value in days_needed
    answer = days_needed[needed - 1]
    print(answer)

# Do not forget to call main at the end of the program
if __name__ == "__main__":
    main()