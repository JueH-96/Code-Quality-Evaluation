def main():
    R = int(input().strip())
    # Next thresholds where the number of ^ increases
    for threshold in (100, 200, 300):
        if R < threshold:
            print(threshold - R)
            return

if __name__ == "__main__":
    main()