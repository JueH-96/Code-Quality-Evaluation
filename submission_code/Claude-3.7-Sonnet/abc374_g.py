def min_strings_ng_list(n, product_names):
    # Convert product names to set for fast lookup
    product_names_set = set(product_names)
    
    # Start with individual product names as strings in the NG list
    strings = list(product_names)
    
    while True:
        merged = False
        for i in range(len(strings)):
            if merged:
                break
            for j in range(len(strings)):
                if i == j:
                    continue
                
                s1, s2 = strings[i], strings[j]
                
                # Try merging s1 and s2 if the last character of s1 matches the first character of s2
                if s1[-1] == s2[0]:
                    merged_string = s1 + s2[1:]
                    # Check if the merged string contains only used product names as length-2 substrings
                    valid = True
                    for k in range(len(merged_string) - 1):
                        if merged_string[k:k+2] not in product_names_set:
                            valid = False
                            break
                    
                    if valid:
                        # Create a new list of strings with the merged string
                        new_strings = []
                        for idx, s in enumerate(strings):
                            if idx != i and idx != j:
                                new_strings.append(s)
                        new_strings.append(merged_string)
                        
                        strings = new_strings
                        merged = True
                        break
        
        if not merged:
            break
    
    return len(strings)

# Read input
n = int(input())
product_names = []
for _ in range(n):
    product_names.append(input())

# Solve the problem and print the answer
print(min_strings_ng_list(n, product_names))