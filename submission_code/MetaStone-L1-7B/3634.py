from collections import defaultdict

def calculateScore(s: str) -> int:
    # Helper function to find the mirror character
    def mirror(c):
        return chr(25 - ord(c))
    
    # Precompute the indices for each character
    char_indices = defaultdict(list)
    for i, c in enumerate(s):
        char_indices[c].append(i)
    
    # Initialize used array to track which indices are used
    used = [False] * len(s)
    # Initialize pointers for each character's list of indices
    pointers = {c: 0 for c in char_indices}
    score = 0
    
    for i in range(len(s)):
        c = s[i]
        mc = mirror(c)
        if mc not in char_indices:
            continue
        
        # Get the list of indices for the mirror character
        indices = char_indices[mc]
        # Find the position where i would be inserted
        pos = bisect.bisect_left(indices, i)
        if pos == 0:
            continue  # No j < i found
        
        # The largest j < i is at pos-1
        j = indices[pos - 1]
        if j >= i:
            continue  # j should be less than i
        
        # Check if j is already used
        if used[j]:
            continue
        
        # Mark j as used
        used[j] = True
        score += (i - j)
        
        # Update the pointer to the next possible j for the mirror character
        pointers[mc] = pos
    
    return score