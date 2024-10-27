const express = require("express");
const router = express.Router();
const Admin = require("../models/Admin.js");
const { body, validationResult } = require("express-validator");
var bcrypt = require("bcryptjs");
var jwt = require("jsonwebtoken");

router.post(
  "/createadmin",
  [
    body("name", "enter the valid name").isLength({ min: 3 }),
    body("email", "enter the valid email").isEmail(),
    body("password", "enter the valid password").isLength({ min: 3 }),
  ],
  async (req, res) => {
    const result = validationResult(req);
    if (!result.isEmpty()) {
      success = false;
      return res.json({ success, errors: result.array() });
    }
    try {
      let admin = await Admin.findOne({ email: req.body.email });
      if (admin) {
        return res.json({
          success: false,
          error: "please enter a unique value for email",
        });
      }
      const salt = await bcrypt.genSaltSync(10);
      const secPass = await bcrypt.hash(req.body.password, salt);

      admin = await Admin.create({
        name: req.body.name,
        email: req.body.email,
        password: secPass,
      });
      /*.then(res.send(req.body))
      .catch(err=>{console.log(err)
        res.json({error:"please enter a unique value for email",message:err.message})
      })*/
      console.log(req.body);
      const data = {
        admin: {
          id: admin.id,
        },
      };
      var token1 = jwt.sign(data, "shhhhh");
      success = true;
      res.json({ success, token1 });
    } catch (error) {
      console.error(error.message);
      success = false;
      res.status(500).send(success, "some error occured");
    }
  }
);
router.post(
    "/loginadmin",
    [
      body("email", "enter the valid email").isEmail(),
      body("password", "password cannot be blank").exists()
    ],
    async (req, res) => {
      const result = validationResult(req);
      if (!result.isEmpty()) {
        success = false;
        return res.status(400).json({ success, errors: result.array() });
      }

      try {
        const { email, password } = req.body;
        let admin = await Admin.findOne({ email: email });
        if (!admin) {
          success = false;
          return res.json({
            success,
            error: "please try to login with correct credentials",
          });
        }
        const passwordCompare = await bcrypt.compare(password, admin.password);
        if (!passwordCompare) {
          success = false;
          return res.json({
            success,
            error: "please try to login with correct credentials",
          });
        }
        const data={
          admin:{
            id:admin.id
          }
         }
        var token1 = jwt.sign(data, "shhhhh");
        success = true;
        res.json({ success, token1 });
      } catch (error) {
        console.error(error.message);
        res.status(500).send("some error occured");
      }
    }
  );
module.exports = router;