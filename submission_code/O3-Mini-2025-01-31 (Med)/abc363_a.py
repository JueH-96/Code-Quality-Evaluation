def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    R = int(input_data[0])
    # Depending on the interval where R belongs, calculate the minimum increase
    if R < 100:
        # If rating is less than 100, then currently one ^, need to reach at least 100 for two ^
        result = 100 - R
    elif R < 200:
        # If rating is between 100 and 199, then need to reach at least 200 for three ^
        result = 200 - R
    elif R < 300:
        # If rating is between 200 and 299, then need to reach at least 300 for four ^
        result = 300 - R
    sys.stdout.write(str(result) + "
")

if __name__ == '__main__':
    main()