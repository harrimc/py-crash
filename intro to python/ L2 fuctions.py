def to_smash(total_candies):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    return total_candies % num_friend

print('How many friends are participating in this candy communism?')
num_friend = int(input('> '))

print('Whats the haul?')
haul = int(input('> '))

candies_each = (haul)//(num_friend)

smooshed = to_smash(haul)

print('Each person gets ', candies_each, ' cadies')
print(smooshed, ' candies are smashed')

