def main():
    N = input().strip()
    # Check the counts of digits '1', '2', and '3'
    if N.count('1') == 1 and N.count('2') == 2 and N.count('3') == 3:
        print("Yes")
    else:
        print("No")

# Do NOT forget to call main()
main()