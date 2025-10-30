def main():
    S = input().strip()
    # Check if rearranging S can form "ABC" by comparing sorted characters
    if sorted(S) == ['A', 'B', 'C']:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()