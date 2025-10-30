def main():
    import sys
    data = sys.stdin.read().strip()
    if not data:
        return
    N = int(data)
    # pi to 100 decimal places as given in the problem
    pi_str = (
        "3."
        "14159265358979323846264338327950288419716939937510"
        "58209749445923078164062862089986280348253421170679"
    )
    # We want up to N decimal places: the dot is at index 1, decimals start at index 2
    result = pi_str[:2 + N]
    sys.stdout.write(result)

if __name__ == "__main__":
    main()