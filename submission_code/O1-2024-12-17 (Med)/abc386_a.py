def main():
    import sys
    
    data = list(map(int, sys.stdin.readline().split()))
    freq = {}
    for val in data:
        freq[val] = freq.get(val, 0) + 1
    
    # Get the frequency counts in descending order
    freq_counts = sorted(freq.values(), reverse=True)
    
    # Possible patterns of 4 cards that can become a full house by adding one card:
    #  - [3, 1] => add one more to the single -> 3,2 => Yes
    #  - [2, 2] => add one more to either pair -> 3,2 => Yes
    # Otherwise => No
    if freq_counts == [3, 1] or freq_counts == [2, 2]:
        print("Yes")
    else:
        print("No")

# Do not forget to call main!
main()