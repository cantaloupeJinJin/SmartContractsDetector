contract christiancoin is DetailedERC20, MintableToken, BurnableToken, PausableToken {

    uint256 public INITIAL_SUPPLY;

    constructor() public DetailedERC20("christiancoin","CNC",18){
        INITIAL_SUPPLY = 2100000000e18;
        mint(msg.sender, INITIAL_SUPPLY);
    }
}