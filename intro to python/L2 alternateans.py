def to_smash(total_candies, num_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between any number friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % num_friends 

print('How many are in this sugar socialism?')
total_fri = int(input('> '))

print('How many are there?')
total_cand = int(input('> '))

cand_fri = total_cand//total_fri



print('Each person gets ',cand_fri, ' peices of candy' )
print(to_smash(total_cand, total_fri), 'candies were smashed')