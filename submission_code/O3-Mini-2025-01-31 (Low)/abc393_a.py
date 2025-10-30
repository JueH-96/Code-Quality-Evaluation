def main():
    import sys
    input_data = sys.stdin.read().split()
    s1, s2 = input_data[0], input_data[1]
    
    # Determine possible culprit sets based on Takahashi (oysters 1 and 2)
    if s1 == "sick":
        takahashi_candidates = {1, 2}
    else:  # "fine"
        takahashi_candidates = {3, 4}
    
    # Determine possible culprit sets based on Aoki (oysters 1 and 3)
    if s2 == "sick":
        aoki_candidates = {1, 3}
    else:  # "fine"
        aoki_candidates = {2, 4}
    
    # The true culprit must be in both sets.
    culprit = (takahashi_candidates & aoki_candidates).pop()
    print(culprit)

if __name__ == '__main__':
    main()