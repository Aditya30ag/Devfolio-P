import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";

export default function Admin(props) {
  const [credentials, setcredential] = useState({ email: "", password: "" });
  const navigate = useNavigate();
  const [content, setcontent] = useState("");
  const password = document.querySelector("#exampleInputPassword1");

  const handleonClick = async (e) => {
    e.preventDefault();
    const url = "http://localhost:5000/api/admin/loginadmin";
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: credentials.email,
        password: credentials.password,
      }),
    });
    const json = await response.json();
    console.log(json);
    if (json.success === true) {
      localStorage.setItem("token1", json.token1);
      navigate("/homeadmin");
      props.showalert();
    } else {
      props.showalert();
      if (json.success1 === false) {
        setcontent(json.error);
      }
      password.style.border = "1px solid red";
      setcontent(json.error);
    }
  };

  const onchange = async (e) => {
    await setcredential({ ...credentials, [e.target.name]: e.target.value });
    setcontent("");
  };

  return (
      <div className="bg-white shadow-md rounded-lg p-8 w-full max-w-md">
        <h3 className="text-3xl font-bold text-center mb-6 text-gray-800">
          Government Portal Login
        </h3>
        <p className="text-sm text-gray-600 text-center mb-4">
          Please enter your credentials to access the admin dashboard.
        </p>
        <form onSubmit={handleonClick} className="space-y-5">
          <div>
            <label
              htmlFor="exampleInputEmail1"
              className="block text-sm font-semibold text-gray-700"
            >
              Admin Id
            </label>
            <input
              type="email"
              id="exampleInputEmail1"
              name="email"
              value={credentials.email} 
              onChange={onchange}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="admin@gov.in"
            />
          </div>

          <div>
            <label
              htmlFor="exampleInputPassword1"
              className="block text-sm font-semibold text-gray-700"
            >
              Password
            </label>
            <input
              type="password"
              id="exampleInputPassword1"
              name="password"
              value={credentials.password}
              onChange={onchange}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
              placeholder="Enter your password"
            />
            {content && (
              <p className="text-red-500 text-sm mt-2">
                {content}
              </p>
            )}
          </div>

          <div className="flex items-center justify-between">
            <button
              type="submit"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Login
            </button>
            <Link
              to="/signupadmin"
              className="text-indigo-600 hover:text-indigo-800 text-sm font-semibold"
            >
              New to this Portal?
            </Link>
          </div>
        </form>
      </div>
  );
}
