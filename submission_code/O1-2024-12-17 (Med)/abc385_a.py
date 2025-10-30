def main():
    A, B, C = map(int, input().split())
    
    # Check if all are equal (which can be put into three groups of equal sums).
    if A == B == C:
        print("Yes")
        return
    
    # Otherwise, check if we can form two groups where one group contains two integers and the other group contains one.
    if A + B == C or A + C == B or B + C == A:
        print("Yes")
    else:
        print("No")

# Do not forget to call main()
main()