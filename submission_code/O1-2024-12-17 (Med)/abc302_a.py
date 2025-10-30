def main():
    import sys
    data = sys.stdin.read().strip().split()
    A, B = map(int, data)
    # Calculate the minimum number of attacks needed
    # Essentially, this is ceil(A / B) which can be computed as (A + B - 1) // B for integers:
    answer = (A + B - 1) // B
    print(answer)

# Don't forget to call the main function
main()