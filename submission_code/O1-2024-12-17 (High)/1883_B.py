def main():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    idx = 1
    
    results = []
    
    for _ in range(t):
        n = int(data[idx]); idx += 1
        k = int(data[idx]); idx += 1
        s = data[idx]; idx += 1
        
        # 1) Count frequencies of letters.
        freq = [0] * 26
        for c in s:
            freq[ord(c) - ord('a')] += 1
        
        # 2) Count how many letters have an odd frequency.
        odd_count = sum(f % 2 for f in freq)
        
        # 3) Compute the base "even removal capacity".
        #    For a letter with freq = x:
        #      - if x is even, we can remove up to x in even increments (0, 2, 4, ..., x)
        #      - if x is odd,  we can remove up to x-1 in even increments
        #    We'll sum these maximum even amounts over all letters.
        sum_even_capacity_base = 0
        even_freq_count = 0  # how many letters have an even frequency (and > 0)
        for f in freq:
            if f > 0 and f % 2 == 0:
                even_freq_count += 1
            if f % 2 == 0:
                sum_even_capacity_base += f
            else:
                sum_even_capacity_base += (f - 1)
        
        # We want to see if we can remove exactly k characters so that
        # the final string of length (n - k) is a palindrome.
        # For a palindrome of even length, we need 0 odd frequencies.
        # For a palindrome of odd length, we need exactly 1 odd frequency.
        
        final_length = n - k
        # Case A) final_length is even => we want 0 odd frequencies
        if final_length % 2 == 0:
            # To get 0 odd frequencies, we must "toggle" (remove an odd number from) all odd letters.
            # That is T = odd_count toggles. Each toggle reduces odd_count by 1 for that letter.
            # Conditions:
            #   1) The parity of T must match the parity of k (so we can remove exactly k in total).
            #      i.e. odd_count % 2 == k % 2
            #   2) T <= k (we can't remove more than k total if each toggle is at least 1 char).
            #   3) leftover = k - T must be <= sum_even_capacity_base (we can remove leftover in even increments).
            
            if odd_count % 2 != k % 2:
                results.append("NO")
                continue
            
            if odd_count > k:
                results.append("NO")
                continue
            
            leftover = k - odd_count
            if leftover <= sum_even_capacity_base:
                results.append("YES")
            else:
                results.append("NO")
        
        # Case B) final_length is odd => we want exactly 1 odd frequency
        else:
            # We need to arrive at final_odd_count = 1.
            # Condition on parity: odd_count must differ mod 2 from k (so T mod 2 == k mod 2 is possible).
            if odd_count % 2 == k % 2:
                results.append("NO")
                continue
            
            # Let O = odd_count.  We want exactly one odd frequency at the end:
            # final_odd_count = O - toggles_odd + toggles_even = 1
            # => toggles_even - toggles_odd = 1 - O
            # => toggles_even = toggles_odd + 1 - O
            # Meanwhile T = toggles_odd + toggles_even <= k, and leftover = k - T >= 0.
            # Also leftover must be <= sum_even_capacity_base - 2*toggles_even,
            #   because toggling an even-freq letter reduces capacity by 2,
            #   while toggling an odd-freq letter does not reduce capacity further.
            
            O = odd_count
            E = even_freq_count
            possible = False
            
            # toggles_odd can only range from 0 to O, but also must ensure toggles_even >= 0.
            # The difference 1 - O might be negative.  
            # We'll simply try all feasible toggles_odd from max(0, O-1) up to O (since
            # if O=0, we might need toggles_odd=0 to get toggles_even=1, etc.).
            
            lower = max(0, O - 1)
            upper = O
            
            for toggles_odd in range(lower, upper + 1):
                toggles_even = toggles_odd + 1 - O
                if toggles_even < 0 or toggles_even > E:
                    continue
                T = toggles_odd + toggles_even
                if T > k:
                    continue
                
                leftover = k - T
                # after toggles_even toggles on even letters, the capacity becomes
                # sum_even_capacity_base - 2 * toggles_even
                if leftover <= sum_even_capacity_base - 2 * toggles_even:
                    possible = True
                    break
            
            results.append("YES" if possible else "NO")
    
    print("
".join(results))

# Don't forget to call main()
if __name__ == "__main__":
    main()