def main():
    A, B, C = map(int, input().split())
    
    # Case 1: All three are equal => can form three groups each with one item
    if A == B == C:
        print("Yes")
        return
    
    # Case 2: Any two sum to the third => can form two groups: two items vs one item
    if A + B == C or A + C == B or B + C == A:
        print("Yes")
        return
    
    # Otherwise, it's not possible to partition into equal-sum groups
    print("No")

# Do not forget to call main()
main()