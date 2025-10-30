# YOUR CODE HERE
s1, s2 = input().split()

# Create a mapping from the observed statuses (Takahashi, Aoki)
# to the type of oyster that must have caused the trouble.
# The keys are tuples (Takahashi's status, Aoki's status).
# The values are the labels of the poisonous oyster.
status_to_poisonous_oyster = {
    ("sick", "sick"): 1,
    ("sick", "fine"): 2,
    ("fine", "sick"): 3,
    ("fine", "fine"): 4
}

# Look up the poisonous oyster type based on the input statuses.
# The problem constraints guarantee that s1 and s2 will be "sick" or "fine",
# so (s1, s2) will always be a valid key in the dictionary.
result_oyster = status_to_poisonous_oyster[(s1, s2)]

print(result_oyster)