pragma solidity ^0.4.25;

import "../libraries/SafeMath.sol";

contract MyContract {

    function currentBlockHash() public view returns(bytes32) {
        return blockhash(block.number);
    }
}