import React from "react"; //{}=구조분해
import { TodoInput, TodoList } from "common";

export default function Todo(){
    return(
        <div>
            <TodoInput/>
            <TodoList/>
        </div>
    )
}