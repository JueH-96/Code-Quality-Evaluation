import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        N = int(data[index])
        K = int(data[index + 1])
        index += 2
        
        cakes = []
        for _ in range(N):
            X = int(data[index])
            Y = int(data[index + 1])
            Z = int(data[index + 2])
            index += 3
            cakes.append((X, Y, Z))
        
        # To maximize the total price, we need to maximize the sum of the maximum pair values.
        # We will sort the cakes by each attribute and try to pair the largest values together.
        # We will calculate the maximum possible pairing price for each attribute and choose the best.
        
        # Sort by X, Y, Z and get the top 2K elements for each
        sorted_by_x = sorted(cakes, key=lambda c: c[0], reverse=True)[:2*K]
        sorted_by_y = sorted(cakes, key=lambda c: c[1], reverse=True)[:2*K]
        sorted_by_z = sorted(cakes, key=lambda c: c[2], reverse=True)[:2*K]
        
        # Calculate the maximum possible sum for each attribute
        max_x_sum = sum(sorted_by_x[i][0] for i in range(2*K))
        max_y_sum = sum(sorted_by_y[i][1] for i in range(2*K))
        max_z_sum = sum(sorted_by_z[i][2] for i in range(2*K))
        
        # The maximum of these sums will give us an upper bound on the best possible pairing
        max_possible_sum = max(max_x_sum, max_y_sum, max_z_sum)
        
        # However, this does not guarantee that we can actually pair them to achieve this sum
        # because the same cake might be in top 2K of multiple attributes.
        # We need to ensure that we can actually form K pairs.
        
        # We will use a greedy approach to actually form pairs and calculate the sum
        # We will use the attribute that gives the highest possible sum as our guide.
        
        if max_possible_sum == max_x_sum:
            sorted_cakes = sorted_by_x
            primary_index = 0
        elif max_possible_sum == max_y_sum:
            sorted_cakes = sorted_by_y
            primary_index = 1
        else:
            sorted_cakes = sorted_by_z
            primary_index = 2
        
        # Now we need to pair the cakes in sorted_cakes to maximize the sum
        # We will use a greedy approach to pair the largest available values
        used = set()
        pairs = []
        
        for cake in sorted_cakes:
            if len(pairs) == K:
                break
            if cake not in used:
                # Find the best pair for this cake
                best_pair = None
                best_value = -1
                for candidate in sorted_cakes:
                    if candidate not in used and candidate != cake:
                        pair_value = max(cake[0] + candidate[0], cake[1] + candidate[1], cake[2] + candidate[2])
                        if pair_value > best_value:
                            best_value = pair_value
                            best_pair = candidate
                if best_pair:
                    pairs.append((cake, best_pair))
                    used.add(cake)
                    used.add(best_pair)
        
        # Calculate the total price of the formed pairs
        total_price = sum(max(pair[0][0] + pair[1][0], pair[0][1] + pair[1][1], pair[0][2] + pair[1][2]) for pair in pairs)
        results.append(total_price)
    
    for result in results:
        print(result)