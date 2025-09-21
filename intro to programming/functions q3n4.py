def get_cost(sqft_walls, sqft_ceiling, sqft_per_gallon, cost_per_gallon) :
    cost = (((sqft_walls + sqft_ceiling)/sqft_per_gallon)*cost_per_gallon)
    return cost 

print('To assess ur decorating costs we need 4 peices of info')

print('1. How many sqft of paint do you need for the walls?')
wall = int(input('> '))

print('2. How many sqft of paint do you need for the ceiling?')
ceil = int(input('> '))

print('3. How much is the paint ur planning on buying per gallon?')
pric = int(input('> '))

tot = get_cost(wall, ceil, 150, pric )
print('your total cost is', tot)