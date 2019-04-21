contract CambodiaImperialHouseFoundation is DetailedERC20, MintableToken, BurnableToken, PausableToken {

    uint256 public INITIAL_SUPPLY;

    constructor() public DetailedERC20("CambodiaImperialHouseFoundation","CIF",18){
        INITIAL_SUPPLY = 10000000000e18;
        mint(msg.sender, INITIAL_SUPPLY);
    }
}