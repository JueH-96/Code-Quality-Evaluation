def main():
    A, B, C = map(int, input().split())
    # If wake-up time C is before sleep time B (no wrap around),
    # he's awake in [C, B).
    if C < B:
        if C <= A < B:
            print("Yes")
        else:
            print("No")
    else:
        # If C > B, his awake period wraps midnight:
        # [C, 24) and [0, B)
        if A >= C or A < B:
            print("Yes")
        else:
            print("No")

# Call main to execute the solution
main()