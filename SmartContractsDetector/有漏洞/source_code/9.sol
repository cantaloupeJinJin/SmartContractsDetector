pragma solidity ^0.4.25;

import "../libraries/SafeMath.sol";

contract SafeSubAndDiv {
    using SafeMath for uint256;

    function sub(uint a, uint b) public returns(uint) {
        return(a.sub(b));
    }

    function div(uint a, uint b) public returns(uint) {
        return(a.div(b));
    }
}