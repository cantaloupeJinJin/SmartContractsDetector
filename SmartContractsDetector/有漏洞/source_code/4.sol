pragma solidity ^0.4.25;

contract dataStorage {
    uint[] public data;

    function writeData(uint[] _data) external {
        for(uint i = data.length; i < _data.length; i++) {
            data.length++;
            data[i]=_data[i];
        }
    }
}
