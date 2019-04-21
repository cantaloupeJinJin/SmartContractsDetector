pragma solidity ^0.4.25;

contract MyContract {

    uint constant BONUS = 500;
    uint constant DELIMITER = 10000;

    function calculateBonus(uint amount) returns (uint) {
        return amount/DELIMITER*BONUS;
    }
}
