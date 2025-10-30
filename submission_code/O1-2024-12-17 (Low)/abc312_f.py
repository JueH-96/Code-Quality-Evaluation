def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    tx_data = list(map(int, input_data[2:]))

    # Separate items by type
    pulltabs = []   # T=0
    regulars = []   # T=1
    openers = []    # T=2

    idx = 0
    for _ in range(N):
        T_i = tx_data[idx]; X_i = tx_data[idx+1]
        idx += 2
        if T_i == 0:
            pulltabs.append(X_i)
        elif T_i == 1:
            regulars.append(X_i)
        else:  # T_i == 2
            openers.append(X_i)

    # Sort descending
    pulltabs.sort(reverse=True)
    regulars.sort(reverse=True)
    openers.sort(reverse=True)

    # Prefix sums (for quick retrieval of top k sums)
    def prefix_sums(arr):
        ps = [0]
        for x in arr:
            ps.append(ps[-1] + x)
        return ps

    ps_pull = prefix_sums(pulltabs)   # sum of top i pulltabs
    ps_reg = prefix_sums(regulars)    # sum of top i regulars
    ps_opn = prefix_sums(openers)     # sum of top i openers usage

    # We try picking k openers, which gives ps_opn[k] total can-opening capacity
    # We can open up to that many regular cans, but cannot exceed M-k in terms of items chosen.
    # Then the rest is filled with pulltab cans, if possible.
    max_happiness = 0
    # We can't pick more openers than the count of openers or M
    max_openers_to_pick = min(len(openers), M)
    
    for k in range(max_openers_to_pick + 1):
        # capacity from openers
        capacity = ps_opn[k]  
        # We can pick at most capacity regular cans, but also limited by how many items remain: M - k
        # and also can't exceed number of available regular cans
        regular_can_count = min(capacity, len(regulars), M - k)
        # pick that many of the top regular cans
        reg_sum = ps_reg[regular_can_count]

        # fill the rest with pulltabs
        # total items so far = k + regular_can_count
        # we can pick up to M - (k + regular_can_count) pulltabs
        pull_count = M - (k + regular_can_count)
        if pull_count < 0:
            continue
        if pull_count > len(pulltabs):
            pull_count = len(pulltabs)
        pull_sum = ps_pull[pull_count]

        hap = reg_sum + pull_sum
        if hap > max_happiness:
            max_happiness = hap

    print(max_happiness)

if __name__ == "__main__":
    main()